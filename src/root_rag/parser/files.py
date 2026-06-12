"""File discovery for code chunking."""
import logging
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# File extensions to include
INCLUDED_EXTENSIONS = {".h", ".hpp", ".hh", ".hxx", ".cxx", ".cpp", ".cc", ".c"}
FAIRSHIP_WORKFLOW_EXTENSIONS = INCLUDED_EXTENSIONS | {".py", ".md", ".C"}
FAIRSHIP_WORKFLOW_DIRS = {"macro", "python", "muonDIS"}

PROJECT_DOCS_EXTENSIONS = {".md", ".json", ".yaml", ".yml", ".txt"}
PROJECT_DOCS_DIRS = {"docs", "reports", "query_packs", "benchmark"}
PROJECT_DOCS_FILES = {"AGENTS.md", "boulder.json"}

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
    "data",
    "artifacts",
}


def _resolve_extension_set(discovery_profile: Optional[str]) -> Set[str]:
    """Resolve extension set for a discovery profile."""
    if discovery_profile == "fairship_workflow":
        return FAIRSHIP_WORKFLOW_EXTENSIONS
    if discovery_profile == "project_docs":
        return PROJECT_DOCS_EXTENSIONS
    return INCLUDED_EXTENSIONS


def discover_text_files(repo_root: Path, discovery_profile: Optional[str] = None) -> List[Path]:
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
    included_extensions = _resolve_extension_set(discovery_profile)
    
    for path in repo_root.rglob("*"):
        # Skip if any directory *within the repo* is excluded. Evaluate parts
        # relative to repo_root so that an excluded name in the checkout location
        # itself (e.g. corpora installed under data/, test fixtures under
        # artifacts/) does not filter out the entire corpus.
        relative_parts = path.relative_to(repo_root).parts
        if any(part in EXCLUDED_DIRS for part in relative_parts):
            continue
        
        if not path.is_file():
            continue

        suffix = path.suffix
        is_readme = path.name.upper().startswith("README")
        is_workflow_dir = any(part in FAIRSHIP_WORKFLOW_DIRS for part in path.parts)
        is_project_docs_dir = any(part in PROJECT_DOCS_DIRS for part in path.parts)
        is_project_docs_file = path.name in PROJECT_DOCS_FILES

        if discovery_profile == "project_docs":
            if is_project_docs_file or is_project_docs_dir:
                if suffix in included_extensions:
                    files.append(path)
            continue

        if suffix in included_extensions:
            files.append(path)
            continue

        # README files are workflow entry points for FairShip discovery.
        if discovery_profile == "fairship_workflow" and is_readme and (
            suffix in {"", ".txt"} or is_workflow_dir
        ):
            files.append(path)
    
    # Sort for deterministic order
    files.sort()
    
    logger.info(
        "Discovered %d text files in %s (profile=%s)",
        len(files),
        repo_root,
        discovery_profile or "default",
    )
    
    return files
