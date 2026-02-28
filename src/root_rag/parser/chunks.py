"""Chunking logic: convert source files to line-based chunks."""
import logging
import re
from pathlib import Path
from typing import List, Tuple

from root_rag.index.schemas import Chunk
from root_rag.parser.files import INCLUDED_EXTENSIONS

logger = logging.getLogger(__name__)

# Pattern to detect Doxygen markers
DOXYGEN_PATTERN = re.compile(r"/\*\*|//!|///<")


def _get_language_from_path(file_path: Path) -> str:
    """Map file extension to language identifier."""
    ext = file_path.suffix.lower()
    ext_to_lang = {
        ".h": "cpp",
        ".hpp": "cpp",
        ".hh": "cpp",
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cxx": "cpp",
    }
    return ext_to_lang.get(ext, "text")


def _get_doc_origin_from_path(file_path: Path) -> str:
    """Determine doc_origin based on file extension."""
    name = file_path.name.lower()
    stem = file_path.stem.lower()
    
    # Header files
    if file_path.suffix in {".h", ".hpp", ".hh"}:
        return "source_header"
    
    # Implementation files
    if file_path.suffix in {".c", ".cc", ".cpp", ".cxx"}:
        return "source_impl"
    
    # Default for unknown
    return "source_impl"


def _read_file_lines(file_path: Path) -> List[str]:
    """Read file and return lines with preserved line endings stripped.
    
    Returns:
        List of lines (without trailing newline)
    
    Raises:
        OSError: If file cannot be read
    """
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with fallback encoding
        text = file_path.read_text(encoding="utf-8", errors="replace")
    
    # Split into lines, preserve content but remove trailing newlines
    lines = text.splitlines(keepends=False)
    return lines


def _has_doxygen_in_chunk(lines: List[str]) -> bool:
    """Check if any line in the chunk has Doxygen markers."""
    chunk_text = "\n".join(lines)
    return bool(DOXYGEN_PATTERN.search(chunk_text))


def chunk_file(
    file_path: Path,
    root_ref: str,
    resolved_commit: str,
    repo_root: Path,
    window_lines: int = 80,
    overlap_lines: int = 10,
) -> List[Chunk]:
    """Chunk a single file using sliding window approach.
    
    Args:
        file_path: Absolute path to file
        root_ref: Root reference (tag/branch/commit)
        resolved_commit: Resolved commit SHA
        repo_root: Repository root (for relative path calculation)
        window_lines: Lines per window/chunk (default 80)
        overlap_lines: Lines of overlap between windows (default 10)
    
    Returns:
        List of Chunk objects for this file, sorted by line number
    
    Raises:
        ValueError: If file cannot be read or is invalid
    """
    file_path = Path(file_path)
    repo_root = Path(repo_root)
    
    # Read file lines
    try:
        lines = _read_file_lines(file_path)
    except Exception as e:
        logger.warning(f"Failed to read file {file_path}: {e}")
        return []
    
    if not lines:
        return []
    
    # Calculate relative path with POSIX separators
    try:
        rel_path = file_path.relative_to(repo_root)
    except ValueError:
        # file_path is not relative to repo_root; use as-is
        rel_path = file_path
    
    file_path_posix = rel_path.as_posix()
    
    language = _get_language_from_path(file_path)
    doc_origin = _get_doc_origin_from_path(file_path)
    
    # Generate chunks using sliding window
    chunks = []
    stride = window_lines - overlap_lines
    total_lines = len(lines)
    
    # Ensure we have at least one window
    if stride <= 0:
        stride = 1
    
    current_start = 0
    while current_start < total_lines:
        current_end = min(current_start + window_lines - 1, total_lines - 1)
        
        # Convert to 1-indexed line numbers
        start_line = current_start + 1
        end_line = current_end + 1
        
        # Extract content for this chunk (lines are 0-indexed)
        chunk_lines = lines[current_start : current_end + 1]
        content = "\n".join(chunk_lines)
        
        # Detect Doxygen
        has_doxygen = _has_doxygen_in_chunk(chunk_lines)
        
        # Create chunk
        chunk = Chunk.from_file_slice(
            file_path=file_path_posix,
            start_line=start_line,
            end_line=end_line,
            content=content,
            root_ref=root_ref,
            resolved_commit=resolved_commit,
            language=language,
            doc_origin=doc_origin,
            has_doxygen=has_doxygen,
        )
        
        chunks.append(chunk)
        
        # Move to next window
        current_start += stride
    
    logger.debug(f"Chunked {file_path_posix}: {len(chunks)} chunks")
    return chunks


def chunk_corpus(
    manifest,
    repo_root: Path,
    window_lines: int = 80,
    overlap_lines: int = 10,
) -> List[Chunk]:
    """Chunk an entire corpus (all files).
    
    Args:
        manifest: Manifest object with root_ref, resolved_commit, local_path
        repo_root: Repository root directory to search
        window_lines: Lines per window (default 80)
        overlap_lines: Lines of overlap (default 10)
    
    Returns:
        Flat list of all chunks, sorted by file path then line number
    """
    from root_rag.parser.files import discover_text_files
    
    repo_root = Path(repo_root)
    
    # Discover all text files
    file_paths = discover_text_files(repo_root)
    
    all_chunks = []
    for file_path in file_paths:
        try:
            chunks = chunk_file(
                file_path=file_path,
                root_ref=manifest.root_ref,
                resolved_commit=manifest.resolved_commit,
                repo_root=repo_root,
                window_lines=window_lines,
                overlap_lines=overlap_lines,
            )
            all_chunks.extend(chunks)
        except Exception as e:
            logger.warning(f"Error chunking {file_path}: {e}")
            continue
    
    logger.info(
        f"Corpus chunked: {len(file_paths)} files, {len(all_chunks)} total chunks"
    )
    
    return all_chunks
