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
    Handles simple unified diffs only.
    """
    import re

    current_file = None
    hunks: dict[str, list[tuple[int, list[str], list[str]]]] = {}

    lines = diff_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # New file
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            if current_file not in hunks:
                hunks[current_file] = []
            i += 1
            continue

        # Skip --- lines
        if line.startswith("--- a/"):
            i += 1
            continue

        # Hunk header
        if line.startswith("@@"):
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if match and current_file:
                old_start = int(match.group(1))
                removes: list[str] = []
                adds: list[str] = []
                i += 1

                while i < len(lines):
                    hline = lines[i]
                    if hline.startswith("@@") or hline.startswith("diff ") or hline.startswith("--- ") or hline.startswith("+++ "):
                        break
                    if hline.startswith("-"):
                        removes.append(hline[1:])
                    elif hline.startswith("+"):
                        adds.append(hline[1:])
                    elif hline.startswith(" "):
                        removes.append(hline[1:])
                        adds.append(hline[1:])
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
                idx = old_start - 1  # 0-indexed
                # Remove old lines
                del file_lines[idx:idx + len(removes)]
                # Insert new lines
                for j, add_line in enumerate(adds):
                    file_lines.insert(idx + j, add_line)

            full_path.write_text("\n".join(file_lines), encoding="utf-8")

        return PatchResult(success=True, message="Patch applied manually")
    except Exception as e:
        return PatchResult(success=False, message=f"Manual apply failed: {e}")


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
        ignore=shutil.ignore_patterns(".git", "__pycache__"),
    )
    console.print(f"[blue]Created working copy at {work_dir}[/blue]")
    return work_dir
