"""
Patch quality metrics: exact match and hunk-level match rate.

Operates on unified diff strings.
"""

import re


def _extract_hunks(diff: str) -> list[str]:
    """Extract individual hunks from a unified diff string.

    A hunk starts with ``@@`` and continues until the next ``@@``,
    the next ``---`` header, or end-of-string.
    """
    hunks: list[str] = []
    current: list[str] = []

    for line in diff.splitlines():
        if line.startswith("@@"):
            if current:
                hunks.append("\n".join(current))
            current = [line]
        elif line.startswith("--- ") or line.startswith("+++ "):
            # File header — flush any current hunk
            if current:
                hunks.append("\n".join(current))
                current = []
        elif current:
            current.append(line)

    if current:
        hunks.append("\n".join(current))

    return hunks


def _normalise(text: str) -> str:
    """Strip whitespace variance for comparison."""
    return re.sub(r"\s+", " ", text).strip()


def exact_match(predicted: str, reference: str) -> float:
    """Return 1.0 if the predicted diff matches the reference exactly (after normalisation), else 0.0."""
    return 1.0 if _normalise(predicted) == _normalise(reference) else 0.0


def hunk_match_rate(predicted: str, reference: str) -> float:
    """Fraction of reference hunks that appear (normalised) in the predicted diff.

    Returns a value between 0.0 and 1.0.
    """
    ref_hunks = _extract_hunks(reference)
    if not ref_hunks:
        return 1.0 if not _extract_hunks(predicted) else 0.0

    pred_normalised = _normalise(predicted)

    matched = 0
    for hunk in ref_hunks:
        # Extract only the added/removed lines for matching
        hunk_lines = []
        for line in hunk.splitlines():
            if line.startswith("+") or line.startswith("-"):
                if not line.startswith("+++") and not line.startswith("---"):
                    hunk_lines.append(line)

        if not hunk_lines:
            matched += 1
            continue

        hunk_content = _normalise("\n".join(hunk_lines))
        if hunk_content and hunk_content in pred_normalised:
            matched += 1

    return matched / len(ref_hunks)
