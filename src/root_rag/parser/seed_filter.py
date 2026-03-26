"""Seed corpus filtering for focused indexing.

Filters ROOT repository files based on seed corpus configuration
to create a minimal, auditable index for MVP.
"""
import logging
from pathlib import Path
from typing import List, Set
import yaml

logger = logging.getLogger(__name__)


def load_seed_corpus_config(config_path: Path) -> dict:
    """Load seed corpus configuration from YAML file.
    
    Args:
        config_path: Path to seed_corpus_root_636.yaml
        
    Returns:
        Dict with corpus configuration
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_seed_corpus_paths(config: dict, repo_root: Path) -> Set[Path]:
    """Extract file paths to index based on seed corpus config.
    
    Args:
        config: Seed corpus configuration dict
        repo_root: Root of ROOT repository checkout
        
    Returns:
        Set of Path objects for files to index
    """
    repo_root = Path(repo_root)
    paths_to_index = set()
    
    classes = config.get("corpus", {}).get("classes", [])
    
    for class_def in classes:
        class_name = class_def.get("name")
        logger.debug(f"Processing class: {class_name}")
        
        # Add headers
        for header_rel in class_def.get("headers", []):
            header_path = repo_root / header_rel
            if header_path.exists():
                paths_to_index.add(header_path)
                logger.debug(f"  Added header: {header_rel}")
            else:
                logger.warning(f"  Header not found: {header_rel}")
        
        # Add source files
        for source_rel in class_def.get("sources", []):
            source_path = repo_root / source_rel
            if source_path.exists():
                paths_to_index.add(source_path)
                logger.debug(f"  Added source: {source_rel}")
            else:
                logger.warning(f"  Source not found: {source_rel}")
    
    logger.info(f"Seed corpus: {len(paths_to_index)} files from {len(classes)} classes")
    return paths_to_index


def filter_files_by_seed_corpus(
    all_files: List[Path],
    seed_corpus_paths: Set[Path],
) -> List[Path]:
    """Filter file list to only include seed corpus files.
    
    Args:
        all_files: List of all discovered files
        seed_corpus_paths: Set of files from seed corpus config
        
    Returns:
        Filtered list of files to index
    """
    # Convert to absolute paths for comparison
    seed_corpus_abs = {p.resolve() for p in seed_corpus_paths}
    
    filtered = []
    for f in all_files:
        if f.resolve() in seed_corpus_abs:
            filtered.append(f)
    
    logger.info(
        f"Filtered {len(all_files)} files → {len(filtered)} seed corpus files"
    )
    
    return filtered
