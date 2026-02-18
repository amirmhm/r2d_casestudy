"""
Diff Guard: validates patches before application.
Rejects patches that are too large or touch too many files.
"""

import re
from dataclasses import dataclass


@dataclass
class DiffStats:
    """Statistics about a unified diff."""
    files_changed: int
    lines_added: int
    lines_removed: int
    total_changed: int
    file_list: list[str]


@dataclass
class GuardResult:
    """Result of the diff guard check."""
    passed: bool
    reason: str
    stats: DiffStats


def parse_diff_stats(diff_text: str) -> DiffStats:
    """
    Parse a unified diff and extract statistics.

    Args:
        diff_text: Raw unified diff text.

    Returns:
        DiffStats with counts.
    """
    files: set[str] = set()
    lines_added = 0
    lines_removed = 0

    for line in diff_text.split("\n"):
        # Track files (handle both plain and quoted paths)
        if line.startswith("+++ b/"):
            files.add(line[6:].strip().strip('"'))
        elif line.startswith('+++ "b/'):
            files.add(line[7:].strip().strip('"'))
        elif line.startswith("--- a/"):
            files.add(line[6:].strip().strip('"'))
        elif line.startswith('--- "a/'):
            files.add(line[7:].strip().strip('"'))
        # Count changes (skip hunk headers and file headers)
        elif line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1

    file_list = sorted(files)
    total = lines_added + lines_removed

    return DiffStats(
        files_changed=len(file_list),
        lines_added=lines_added,
        lines_removed=lines_removed,
        total_changed=total,
        file_list=file_list,
    )


def check_diff(
    diff_text: str,
    max_lines: int = 250,
    max_files: int = 6,
    override: bool = False,
) -> GuardResult:
    """
    Check if a diff passes the guard constraints.

    Args:
        diff_text: Raw unified diff text.
        max_lines: Maximum total changed lines allowed.
        max_files: Maximum files touched allowed.
        override: If True, skip the guard (always pass).

    Returns:
        GuardResult indicating pass/fail and reason.
    """
    stats = parse_diff_stats(diff_text)

    if override:
        return GuardResult(
            passed=True,
            reason="Guard overridden",
            stats=stats,
        )

    if stats.total_changed > max_lines:
        return GuardResult(
            passed=False,
            reason=f"Patch too large: {stats.total_changed} changed lines (max {max_lines})",
            stats=stats,
        )

    if stats.files_changed > max_files:
        return GuardResult(
            passed=False,
            reason=f"Too many files: {stats.files_changed} files (max {max_files})",
            stats=stats,
        )

    if stats.total_changed == 0:
        return GuardResult(
            passed=False,
            reason="Empty patch: no changes detected",
            stats=stats,
        )

    return GuardResult(
        passed=True,
        reason=f"OK: {stats.total_changed} lines in {stats.files_changed} files",
        stats=stats,
    )


def extract_diff_from_response(response: str) -> str:
    """
    Extract unified diff from an LLM response.
    Handles cases where the LLM wraps the diff in markdown code blocks.
    """
    # Try to extract from code blocks first
    code_block_pattern = r'```(?:diff|patch)?\s*\n(.*?)\n```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # If no code blocks, look for diff-like content
    lines = response.strip().split("\n")
    diff_lines: list[str] = []
    in_diff = False

    for line in lines:
        if line.startswith("--- a/") or line.startswith("diff --git"):
            in_diff = True
        if in_diff:
            diff_lines.append(line)

    if diff_lines:
        return "\n".join(diff_lines)

    # Last resort: return the whole response
    return response.strip()
