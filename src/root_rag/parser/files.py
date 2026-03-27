"""File discovery for code chunking."""
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# File extensions to include
INCLUDED_EXTENSIONS = {".h", ".hpp", ".hh", ".cxx", ".cpp", ".cc", ".c"}

# Directories to exclude (case-insensitive on some systems, but match exactly)
EXCLUDED_DIRS = {
    "build",
    ".git",
    ".github",
    "external",
    "qa",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
}


def discover_text_files(repo_root: Path) -> List[Path]:
    """Discover textcode files in repository.
    
    Args:
        repo_root: Root directory to search
    
    Returns:
        Sorted list of file paths (relative to repo_root)
    
    Filtering rules:
        - Include only: .h, .hpp, .hh, .cxx, .cpp, .cc, .c
        - Exclude directories: build/, .git/, external/, qa/, etc.
        - Sorted for deterministic ordering
    """
    repo_root = Path(repo_root)
    
    if not repo_root.is_dir():
        raise ValueError(f"repo_root must be a directory: {repo_root}")
    
    files = []
    
    for path in repo_root.rglob("*"):
        # Skip if any parent directory is excluded
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        
        # Only include files with target extensions
        if path.is_file() and path.suffix in INCLUDED_EXTENSIONS:
            files.append(path)
    
    # Sort for deterministic order
    files.sort()
    
    logger.info(f"Discovered {len(files)} text files in {repo_root}")
    
    return files
