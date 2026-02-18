"""
Structured output handling and chain-of-thought prompting.

Provides Pydantic models for LLM responses, JSON extraction utilities,
and chain-of-thought prompt templates.
"""

import json
import re
from typing import Optional, TypeVar, Type

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class PatchAnalysis(BaseModel):
    """Pre-patch analysis: what files to change and why."""

    files_to_modify: list[str] = []
    reasoning: str = ""
    approach: str = ""
    risks: list[str] = []


class PatchResponse(BaseModel):
    """Structured response containing a unified diff patch."""

    diff: str = ""
    explanation: str = ""
    files_modified: list[str] = []
    confidence: float = 0.0

    @field_validator("diff")
    @classmethod
    def validate_diff(cls, v: str) -> str:
        """Basic validation that the diff looks like a unified diff."""
        v = v.strip()
        if v and not ("---" in v or "+++" in v or "@@" in v):
            # Not a valid unified diff — could be raw code
            pass  # allow through; diff_guard will catch real issues
        return v


class AnalysisResponse(BaseModel):
    """Structured analysis of a code change or task result."""

    success: bool = False
    issues: list[str] = []
    suggestions: list[str] = []
    summary: str = ""


T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# JSON Extraction
# ---------------------------------------------------------------------------

def extract_json_from_response(text: str) -> Optional[dict]:
    """Extract a JSON object from an LLM response using multiple strategies.

    Strategies (tried in order):
      1. Direct ``json.loads`` on the full text.
      2. Extract from a ```json code block.
      3. Brace-matching: find the outermost ``{ ... }``.

    Returns ``None`` if no valid JSON can be extracted.
    """
    text = text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Code block extraction  (```json ... ``` or ``` ... ```)
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Brace matching — find outermost { }
    depth = 0
    start_idx: Optional[int] = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start_idx is not None:
                candidate = text[start_idx : i + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    start_idx = None

    return None


def parse_structured_response(
    text: str,
    model_class: Type[T],
) -> Optional[T]:
    """Parse LLM text into a Pydantic model via JSON extraction.

    Returns ``None`` if extraction or validation fails.
    """
    data = extract_json_from_response(text)
    if data is None:
        return None
    try:
        return model_class.model_validate(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# JSON-mode prompt builder
# ---------------------------------------------------------------------------

def build_json_mode_prompt(
    model_class: Type[BaseModel],
    task_description: str,
) -> str:
    """Build a prompt that instructs the LLM to respond in JSON matching the schema.

    Args:
        model_class: The Pydantic model whose schema the LLM should follow.
        task_description: What the LLM should do.

    Returns:
        A formatted prompt string.
    """
    schema = model_class.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    return (
        f"{task_description}\n\n"
        f"Respond with a JSON object matching this schema:\n"
        f"```json\n{schema_str}\n```\n\n"
        f"Return ONLY valid JSON. No extra text."
    )


# ---------------------------------------------------------------------------
# Chain-of-Thought Templates
# ---------------------------------------------------------------------------

COT_PATCH_TEMPLATE = """You are an expert TypeScript developer generating minimal unified diff patches.

Think step by step before writing the patch:

1. **Understand** the task requirements and acceptance criteria.
2. **Identify** which files and functions need to be modified.
3. **Plan** the minimal set of changes required.
4. **Consider** edge cases, type safety, and existing patterns.
5. **Generate** the unified diff patch.

TASK: {task_title}

REQUEST:
{user_request}

ACCEPTANCE CRITERIA:
{acceptance_criteria}

SUGGESTED FILES:
{suggested_files}

RELEVANT CODE:
{code_context}

Now think step by step and generate the patch.
Respond with a JSON object:
{{
  "reasoning": "Your step-by-step analysis...",
  "files_to_modify": ["file1.ts", "file2.ts"],
  "diff": "--- a/file\\n+++ b/file\\n@@ ... @@\\n ...",
  "confidence": 0.9
}}
"""

COT_ANALYSIS_TEMPLATE = """Analyze the following code change and evaluate its quality.

Think step by step:

1. **Read** the original task requirements.
2. **Examine** the generated diff carefully.
3. **Check** if all acceptance criteria are met.
4. **Identify** any issues (type errors, mutations, missing exports, etc.).
5. **Suggest** improvements if needed.

TASK: {task_title}

REQUIREMENTS:
{requirements}

GENERATED DIFF:
{diff}

Respond with a JSON object:
{{
  "success": true/false,
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1"],
  "summary": "Overall assessment..."
}}
"""


def build_cot_patch_prompt(
    task_title: str,
    user_request: str,
    acceptance_criteria: list[str],
    suggested_files: list[str],
    code_context: str,
) -> list[dict[str, str]]:
    """Build a chain-of-thought prompt for patch generation.

    Returns a message list suitable for chat completion.
    """
    criteria_text = "\n".join(f"- {c}" for c in acceptance_criteria)
    files_text = "\n".join(f"- {f}" for f in suggested_files)

    user_content = COT_PATCH_TEMPLATE.format(
        task_title=task_title,
        user_request=user_request,
        acceptance_criteria=criteria_text,
        suggested_files=files_text,
        code_context=code_context,
    )

    return [
        {
            "role": "system",
            "content": (
                "You are an expert TypeScript developer. "
                "Think step by step, then output a JSON object with your reasoning and the unified diff."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def build_cot_analysis_prompt(
    task_title: str,
    requirements: str,
    diff: str,
) -> list[dict[str, str]]:
    """Build a chain-of-thought prompt for diff quality analysis.

    Returns a message list suitable for chat completion.
    """
    user_content = COT_ANALYSIS_TEMPLATE.format(
        task_title=task_title,
        requirements=requirements,
        diff=diff,
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a code review expert. "
                "Think step by step, then output a JSON object with your analysis."
            ),
        },
        {"role": "user", "content": user_content},
    ]
