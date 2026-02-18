"""
Codebase indexer for RAG.
Builds a manifest of source files with content chunks.

BASELINE: Fixed-window chunking is provided and working.
TODO: Implement AST-aware chunking for Phase 1.
"""

import hashlib
import json
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from rich.console import Console

console = Console()


@dataclass
class CodeChunk:
    """A chunk of source code with metadata."""
    file_path: str
    start_line: int
    end_line: int
    content: str
    content_hash: str
    language: str = "typescript"
    symbols: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    chunk_type: str = "fixed"  # "fixed", "ast", "hybrid"


@dataclass
class CodebaseIndex:
    """Index of all code chunks in the codebase."""
    chunks: list[CodeChunk] = field(default_factory=list)
    file_hashes: dict[str, str] = field(default_factory=dict)
    version: str = "1.0.0"


def hash_content(content: str) -> str:
    """SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def extract_symbols(content: str) -> list[str]:
    """Extract function/class/interface names from TypeScript code."""
    symbols: list[str] = []
    for line in content.split("\n"):
        stripped = line.strip()
        for keyword in ("export function ", "export const ", "function ", "interface ", "type ", "class "):
            if stripped.startswith(keyword):
                rest = stripped[len(keyword):].split("(")[0].split("{")[0].split("<")[0].split(":")[0].strip()
                if rest:
                    symbols.append(rest)
    return symbols


def extract_imports(content: str) -> list[str]:
    """Extract import paths from TypeScript/JavaScript code."""
    imports: list[str] = []
    for line in content.split("\n"):
        stripped = line.strip()
        # Match: import { ... } from "path"  or  import ... from "path"
        match = re.search(r'from\s+["\']([^"\']+)["\']', stripped)
        if match:
            imports.append(match.group(1))
        # Match: import "path"
        elif stripped.startswith("import ") and ("'" in stripped or '"' in stripped):
            match = re.search(r'import\s+["\']([^"\']+)["\']', stripped)
            if match:
                imports.append(match.group(1))
    return imports


def _find_ast_boundaries(lines: list[str]) -> list[tuple[int, int, str]]:
    """Find function/class/interface boundaries in TypeScript code.
    Returns list of (start_line_0indexed, end_line_0indexed, symbol_name).

    Uses brace-counting to detect boundaries of exported/non-exported
    functions, classes, and interfaces.
    """
    boundaries: list[tuple[int, int, str]] = []
    # Patterns that start a top-level block
    block_re = re.compile(
        r"^(?:export\s+)?(?:function|const|class|interface|type|enum)\s+"
        r"([A-Za-z_$][A-Za-z0-9_$]*)"
    )

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        match = block_re.match(stripped)
        if match:
            symbol_name = match.group(1)
            start = i

            # Count braces to find the end of the block
            brace_depth = 0
            found_open = False
            j = i
            while j < len(lines):
                for ch in lines[j]:
                    if ch == "{":
                        brace_depth += 1
                        found_open = True
                    elif ch == "}":
                        brace_depth -= 1

                if found_open and brace_depth <= 0:
                    boundaries.append((start, j, symbol_name))
                    i = j + 1
                    break
                j += 1
            else:
                # Reached end of file without closing brace — single-line or type alias
                boundaries.append((start, min(i, len(lines) - 1), symbol_name))
                i += 1
        else:
            i += 1

    return boundaries


def chunk_file(
    file_path: Path,
    base_dir: Path,
    chunk_size: int = 60,
    strategy: str = "fixed",
) -> list[CodeChunk]:
    """
    Split a file into chunks using the specified strategy.

    Args:
        file_path: Path to the source file.
        base_dir: Base directory for relative path computation.
        chunk_size: Lines per chunk (for fixed strategy).
        strategy: "fixed" (window-based), "ast" (function boundaries), "hybrid" (both).

    Returns:
        List of CodeChunk objects.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    lines = content.split("\n")
    rel_path = str(file_path.relative_to(base_dir)).replace("\\", "/")
    lang = "typescript" if file_path.suffix in (".ts", ".tsx") else "javascript"
    file_imports = extract_imports(content)

    if strategy == "ast":
        return _chunk_ast(lines, rel_path, lang, file_imports)
    elif strategy == "hybrid":
        fixed = _chunk_fixed(lines, rel_path, lang, file_imports, chunk_size)
        ast = _chunk_ast(lines, rel_path, lang, file_imports)
        return fixed + ast
    else:
        return _chunk_fixed(lines, rel_path, lang, file_imports, chunk_size)


# ============================================================================
# BASELINE: Fixed-window chunking (PROVIDED — do not modify)
# ============================================================================

def _chunk_fixed(
    lines: list[str],
    rel_path: str,
    lang: str,
    file_imports: list[str],
    chunk_size: int = 60,
) -> list[CodeChunk]:
    """Fixed-window chunking with overlap."""
    content = "\n".join(lines)
    chunks: list[CodeChunk] = []

    if len(lines) <= chunk_size:
        chunks.append(CodeChunk(
            file_path=rel_path,
            start_line=1,
            end_line=len(lines),
            content=content,
            content_hash=hash_content(content),
            language=lang,
            symbols=extract_symbols(content),
            imports=file_imports,
            chunk_type="fixed",
        ))
        return chunks

    overlap = 10
    start = 0
    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        chunk_content = "\n".join(lines[start:end])
        chunks.append(CodeChunk(
            file_path=rel_path,
            start_line=start + 1,
            end_line=end,
            content=chunk_content,
            content_hash=hash_content(chunk_content),
            language=lang,
            symbols=extract_symbols(chunk_content),
            imports=file_imports,
            chunk_type="fixed",
        ))
        start = end - overlap
        if start + overlap >= len(lines):
            break

    return chunks


# ============================================================================
# TODO: Implement AST-aware chunking
# ============================================================================

def _chunk_ast(
    lines: list[str],
    rel_path: str,
    lang: str,
    file_imports: list[str],
) -> list[CodeChunk]:
    """AST-aware chunking at function/class boundaries.

    Falls back to the whole file as a single chunk when no boundaries
    are detected (e.g. files that only contain imports/type aliases).
    """
    boundaries = _find_ast_boundaries(lines)

    if not boundaries:
        # No detectable blocks — return whole file as one chunk
        content = "\n".join(lines)
        return [
            CodeChunk(
                file_path=rel_path,
                start_line=1,
                end_line=len(lines),
                content=content,
                content_hash=hash_content(content),
                language=lang,
                symbols=extract_symbols(content),
                imports=file_imports,
                chunk_type="ast",
            )
        ]

    chunks: list[CodeChunk] = []

    # Include leading imports/comments before the first boundary
    first_start = boundaries[0][0]
    if first_start > 0:
        preamble = "\n".join(lines[:first_start])
        if preamble.strip():
            chunks.append(
                CodeChunk(
                    file_path=rel_path,
                    start_line=1,
                    end_line=first_start,
                    content=preamble,
                    content_hash=hash_content(preamble),
                    language=lang,
                    symbols=extract_symbols(preamble),
                    imports=file_imports,
                    chunk_type="ast",
                )
            )

    for start, end, _symbol in boundaries:
        chunk_content = "\n".join(lines[start : end + 1])
        chunks.append(
            CodeChunk(
                file_path=rel_path,
                start_line=start + 1,
                end_line=end + 1,
                content=chunk_content,
                content_hash=hash_content(chunk_content),
                language=lang,
                symbols=extract_symbols(chunk_content),
                imports=file_imports,
                chunk_type="ast",
            )
        )

    return chunks


# ============================================================================
# Index management (PROVIDED — do not modify)
# ============================================================================

def index_codebase(
    source_dir: Path,
    extensions: Optional[list[str]] = None,
    chunk_size: int = 60,
    strategy: str = "fixed",
) -> CodebaseIndex:
    """
    Index all source files in a directory.

    Args:
        source_dir: Root directory to index.
        extensions: File extensions to include (default: .ts, .tsx, .js, .mjs).
        chunk_size: Lines per chunk (for fixed strategy).
        strategy: Chunking strategy: "fixed", "ast", or "hybrid".

    Returns:
        CodebaseIndex with all chunks.
    """
    if extensions is None:
        extensions = [".ts", ".tsx", ".js", ".mjs"]

    index = CodebaseIndex()

    if not source_dir.exists():
        console.print(f"[red]Source directory not found: {source_dir}[/red]")
        return index

    source_files = []
    for ext in extensions:
        source_files.extend(source_dir.rglob(f"*{ext}"))

    # Filter out node_modules and dist
    source_files = [
        f for f in source_files
        if "node_modules" not in str(f) and "dist" not in str(f)
    ]

    console.print(f"[blue]Indexing {len(source_files)} files ({strategy} strategy) from {source_dir}[/blue]")

    for file_path in sorted(source_files):
        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        rel_path = str(file_path.relative_to(source_dir)).replace("\\", "/")
        index.file_hashes[rel_path] = hash_content(content)

        chunks = chunk_file(file_path, source_dir, chunk_size, strategy)
        index.chunks.extend(chunks)

    console.print(f"[green]Indexed {len(index.chunks)} chunks from {len(index.file_hashes)} files[/green]")
    return index


def save_index(index: CodebaseIndex, output_path: Path) -> None:
    """Save index to JSON file."""
    data = {
        "version": index.version,
        "file_hashes": index.file_hashes,
        "chunks": [asdict(c) for c in index.chunks],
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    console.print(f"[green]Index saved to {output_path}[/green]")


def load_index(index_path: Path) -> CodebaseIndex:
    """Load index from JSON file."""
    data = json.loads(index_path.read_text(encoding="utf-8"))
    index = CodebaseIndex(
        version=data.get("version", "1.0.0"),
        file_hashes=data.get("file_hashes", {}),
        chunks=[
            CodeChunk(**chunk) for chunk in data.get("chunks", [])
        ],
    )
    return index
