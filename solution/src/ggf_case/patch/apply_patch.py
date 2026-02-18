"""
Patch application using git apply.
Applies unified diffs to a working copy of the codebase.
"""

import subprocess
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass

from rich.console import Console

console = Console()


@dataclass
class PatchResult:
    """Result of applying a patch."""
    success: bool
    message: str
    patch_file: str = ""


def apply_patch(
    diff_text: str,
    working_dir: Path,
    dry_run: bool = False,
) -> PatchResult:
    """
    Apply a unified diff to a working directory using git apply.

    Args:
        diff_text: The unified diff text.
        working_dir: Directory to apply the patch in.
        dry_run: If True, only check if the patch applies cleanly.

    Returns:
        PatchResult with success status and message.
    """
    if not diff_text.strip():
        return PatchResult(success=False, message="Empty patch")

    # Write diff to temp file
    try:
        patch_file = working_dir / ".tmp_patch.diff"
        patch_file.write_text(diff_text, encoding="utf-8")
    except OSError as e:
        return PatchResult(success=False, message=f"Failed to write patch file: {e}")

    try:
        # Check if git is available
        cmd = ["git", "apply"]
        if dry_run:
            cmd.append("--check")
        cmd.append(str(patch_file))

        result = subprocess.run(
            cmd,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            action = "would apply" if dry_run else "applied"
            return PatchResult(
                success=True,
                message=f"Patch {action} successfully",
                patch_file=str(patch_file),
            )
        else:
            # git apply failed — fall back to manual apply
            manual_result = _manual_apply(diff_text, working_dir, dry_run)
            if manual_result.success:
                return manual_result
            error = result.stderr.strip() or result.stdout.strip()
            return PatchResult(
                success=False,
                message=f"git apply failed: {error}",
                patch_file=str(patch_file),
            )

    except FileNotFoundError:
        # git not available, try manual apply
        return _manual_apply(diff_text, working_dir, dry_run)
    except subprocess.TimeoutExpired:
        return PatchResult(success=False, message="Patch application timed out")
    except Exception as e:
        return PatchResult(success=False, message=f"Unexpected error: {e}")
    finally:
        # Clean up temp file
        try:
            if "patch_file" in locals():
                patch_file.unlink(missing_ok=True)
        except OSError:
            pass


def _manual_apply(
    diff_text: str,
    working_dir: Path,
    dry_run: bool = False,
) -> PatchResult:
    """
    Fallback manual patch application for environments without git.
    Handles simple unified diffs, including malformed ones from LLMs.
    """
    import re

    current_file = None
    hunks: dict[str, list[tuple[int, list[str], list[str]]]] = {}

    lines = diff_text.split("\n")
    i = 0

    def _is_header(ln: str) -> bool:
        s = ln.lstrip()
        return (s.startswith("diff --git")
                or s.startswith("--- a/") or s.startswith("--- /dev/null")
                or s.startswith("+++ b/") or s.startswith("+++ /dev/null")
                or s.startswith("@@ "))

    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # New file — handle both clean and space-prefixed headers
        if stripped.startswith("+++ b/"):
            current_file = stripped[6:].strip().strip('"')
            if current_file not in hunks:
                hunks[current_file] = []
            i += 1
            continue

        # Skip --- lines and diff --git lines
        if stripped.startswith("--- a/") or stripped.startswith("--- /dev/null") or stripped.startswith("diff --git"):
            i += 1
            continue

        # Hunk header
        if stripped.startswith("@@"):
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", stripped)
            if match and current_file:
                old_start = int(match.group(1))
                removes: list[str] = []
                adds: list[str] = []
                i += 1

                while i < len(lines):
                    hline = lines[i]
                    hstripped = hline.lstrip()

                    # Stop at next header
                    if _is_header(hline):
                        break

                    if hline.startswith("-"):
                        removes.append(hline[1:])
                    elif hline.startswith("+"):
                        adds.append(hline[1:])
                    elif hline.startswith(" "):
                        # Could be context line or LLM-mangled +/- line
                        if len(hline) > 1 and hline[1] == "+" and not hline.startswith(" +++"):
                            # " +" → added line
                            adds.append(hline[2:])
                        elif len(hline) > 1 and hline[1] == "-" and not hline.startswith(" ---"):
                            # " -" → removed line
                            removes.append(hline[2:])
                        else:
                            # True context line
                            removes.append(hline[1:])
                            adds.append(hline[1:])
                    elif hline == "":
                        # Empty line inside a hunk = context line with empty content
                        removes.append("")
                        adds.append("")
                    else:
                        # Unknown line — skip it
                        pass
                    i += 1

                hunks[current_file].append((old_start, removes, adds))
                continue

        i += 1

    if not hunks:
        return PatchResult(success=False, message="No valid hunks found in patch")

    if dry_run:
        return PatchResult(success=True, message="Dry run: patch appears valid")

    # Apply hunks
    try:
        for file_path, file_hunks in hunks.items():
            full_path = working_dir / file_path

            if full_path.exists():
                content = full_path.read_text(encoding="utf-8")
                file_lines = content.split("\n")
            else:
                # New file
                full_path.parent.mkdir(parents=True, exist_ok=True)
                file_lines = []

            # Apply hunks in reverse order to preserve line numbers
            for old_start, removes, adds in reversed(file_hunks):
                idx = max(0, old_start - 1)  # 0-indexed, clamp for new files

                # New file (empty): just write the adds directly
                if not file_lines and adds:
                    file_lines = list(adds)
                    continue

                if removes:
                    # Fuzzy matching: if exact match fails at idx, search nearby
                    matched_idx = _fuzzy_find(file_lines, removes, idx)
                    if matched_idx is not None:
                        idx = matched_idx
                        # Remove old lines and insert new
                        del file_lines[idx:idx + len(removes)]
                        for j, add_line in enumerate(adds):
                            file_lines.insert(idx + j, add_line)
                    else:
                        # Context doesn't match — try partial context matching
                        # Look for a unique anchor line from the context
                        new_lines = [a for a in adds if a.rstrip() not in {r.rstrip() for r in removes}]
                        if new_lines:
                            anchor_idx = _find_anchor(file_lines, removes, idx)
                            if anchor_idx is not None:
                                insert_at = anchor_idx
                                for j, add_line in enumerate(new_lines):
                                    file_lines.insert(insert_at + j, add_line)
                            # else: skip hunk — don't corrupt the file
                else:
                    # Pure addition — insert at position (or append for new file)
                    insert_at = max(0, min(idx, len(file_lines)))
                    for j, add_line in enumerate(adds):
                        file_lines.insert(insert_at + j, add_line)

            full_path.write_text("\n".join(file_lines), encoding="utf-8")

        return PatchResult(success=True, message="Patch applied manually")
    except Exception as e:
        return PatchResult(success=False, message=f"Manual apply failed: {e}")


def _fuzzy_find(
    file_lines: list[str],
    removes: list[str],
    expected_idx: int,
    search_range: int = 50,
) -> int | None:
    """Find the best matching position for a hunk's removed lines."""
    if not removes:
        return expected_idx

    # Try exact position first
    if _lines_match(file_lines, removes, expected_idx):
        return expected_idx

    # Search nearby
    for offset in range(1, search_range + 1):
        for candidate in (expected_idx + offset, expected_idx - offset):
            if 0 <= candidate <= len(file_lines) - len(removes):
                if _lines_match(file_lines, removes, candidate):
                    return candidate

    return None


def _lines_match(file_lines: list[str], removes: list[str], idx: int) -> bool:
    """Check if removed lines match file content at the given index."""
    if idx < 0 or idx + len(removes) > len(file_lines):
        return False
    for j, rm in enumerate(removes):
        if file_lines[idx + j].rstrip() != rm.rstrip():
            return False
    return True


def _find_anchor(
    file_lines: list[str],
    removes: list[str],
    expected_idx: int,
) -> int | None:
    """Find an insertion point using the last non-blank context line as anchor."""
    # Find the last non-trivial context line
    for rm in reversed(removes):
        stripped = rm.rstrip()
        if stripped and stripped not in ("", "{", "}", ");", "};"):
            # Search the file for this line
            for i, fl in enumerate(file_lines):
                if fl.rstrip() == stripped:
                    return i + 1  # Insert after the anchor
    return None


def create_working_copy(source_dir: Path, target_dir: Path) -> Path:
    """
    Create a working copy of the source directory.

    Args:
        source_dir: Source directory to copy.
        target_dir: Target parent directory.

    Returns:
        Path to the working copy.
    """
    work_dir = target_dir / "work"
    if work_dir.exists():
        shutil.rmtree(work_dir)

    shutil.copytree(
        source_dir, work_dir, dirs_exist_ok=False,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git", "__pycache__"),
    )
    console.print(f"[blue]Created working copy at {work_dir}[/blue]")
    return work_dir
