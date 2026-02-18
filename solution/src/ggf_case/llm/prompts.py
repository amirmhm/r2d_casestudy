"""
Prompt templates for patch generation.
"""

SYSTEM_PROMPT = """You are an expert TypeScript developer. Your job is to generate minimal unified diff patches to modify a TypeScript game codebase.

RULES:
1. Generate ONLY valid unified diff format (--- a/file, +++ b/file, @@ hunks)
2. Produce MINIMAL changes — do not rewrite entire files
3. Preserve existing formatting, indentation, and style
4. Do not rename files unless explicitly required
5. Do not add unnecessary imports or comments
6. Ensure the patched code compiles with TypeScript strict mode
7. Each patch must be self-contained and independently applicable
8. Use the exact file paths shown in the context

OUTPUT FORMAT:
Return ONLY the unified diff. No explanations, no markdown code blocks, no extra text.
Start directly with --- a/ and end with the last hunk.
"""

PATCH_PROMPT_TEMPLATE = """TASK: {task_title}

USER REQUEST:
{user_request}

ACCEPTANCE CRITERIA:
{acceptance_criteria}

SUGGESTED FILES TO MODIFY:
{suggested_files}

RELEVANT CODE CONTEXT:
{code_context}

Generate a minimal unified diff patch that satisfies all acceptance criteria.
Remember: output ONLY the raw unified diff, nothing else."""


def build_patch_prompt(
    task_title: str,
    user_request: str,
    acceptance_criteria: list[str],
    suggested_files: list[str],
    code_context: str,
) -> list[dict[str, str]]:
    """
    Build the full message list for patch generation.

    Returns:
        List of message dicts for chat completion.
    """
    criteria_text = "\n".join(f"- {c}" for c in acceptance_criteria)
    files_text = "\n".join(f"- {f}" for f in suggested_files)

    user_content = PATCH_PROMPT_TEMPLATE.format(
        task_title=task_title,
        user_request=user_request,
        acceptance_criteria=criteria_text,
        suggested_files=files_text,
        code_context=code_context,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
