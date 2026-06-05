"""Index locator: resolve indices by ID or root_ref."""
import logging
from pathlib import Path
from typing import Optional

from root_rag.core.errors import IndexNotFoundError
from root_rag.index.schemas import IndexManifest

logger = logging.getLogger(__name__)
def _resolve_manifest_paths(manifest: IndexManifest, manifest_file: Path) -> IndexManifest:
    """Resolve relative manifest artifact paths.

    If a path is relative, it is first checked relative to the current working 
    directory. If not found, it is resolved against the manifest directory.
    """
    manifest_dir = manifest_file.parent

    # Resolve chunks_path
    chunks_path = Path(manifest.chunks_path)
    if not chunks_path.is_absolute():
        # Try relative to CWD first
        if chunks_path.exists():
            manifest.chunks_path = str(chunks_path.resolve())
        else:
            # Fallback to manifest_dir
            manifest.chunks_path = str((manifest_dir / chunks_path).resolve())

    # Resolve fts_db_path
    fts_db_path = Path(manifest.fts_db_path)
    if not fts_db_path.is_absolute():
        # Try relative to CWD first
        if fts_db_path.exists():
            manifest.fts_db_path = str(fts_db_path.resolve())
        else:
            # Fallback to manifest_dir
            manifest.fts_db_path = str((manifest_dir / fts_db_path).resolve())

    # Resolve semantic_manifest_path
    if manifest.semantic_manifest_path:
        semantic_manifest_path = Path(manifest.semantic_manifest_path)
        if not semantic_manifest_path.is_absolute():
            if semantic_manifest_path.exists():
                manifest.semantic_manifest_path = str(semantic_manifest_path.resolve())
            else:
                manifest.semantic_manifest_path = str((manifest_dir / semantic_manifest_path).resolve())

    return manifest


def resolve_index(
    indexes_root: Path,
    root_ref: Optional[str] = None,
    index_id: Optional[str] = None,
) -> IndexManifest:
    """Resolve an index by explicit ID or by root_ref (picks latest).
    
    Args:
        indexes_root: Root directory containing index subdirectories
        root_ref: ROOT version reference (tag/branch/commit) - optional
        index_id: Explicit index ID - takes precedence over root_ref
        
    Returns:
        IndexManifest object loaded from the resolved index directory
        
    Raises:
        IndexNotFoundError: If index cannot be found or resolved
    """
    indexes_root = Path(indexes_root)
    
    # If explicit index_id provided, use it
    if index_id:
        index_dir = indexes_root / index_id
        manifest_file = index_dir / "index_manifest.json"
        
        if not manifest_file.exists():
            raise IndexNotFoundError(f"Index not found: {index_id}")
        
        logger.info(f"Resolved index by ID: {index_id}")
        return _resolve_manifest_paths(IndexManifest.load(manifest_file), manifest_file)
    
    # Otherwise resolve by root_ref (pick latest timestamp)
    if not root_ref:
        raise IndexNotFoundError("Must provide either --index-id or --root-ref")
    
    # Find all index directories matching the root_ref pattern
    # Pattern: {corpus_id}__{timestamp} where corpus_id contains root_ref
    matching_indices = []
    
    if not indexes_root.exists():
        raise IndexNotFoundError(f"Indexes root directory not found: {indexes_root}")
    
    for index_dir in indexes_root.iterdir():
        if not index_dir.is_dir():
            continue
        
        manifest_file = index_dir / "index_manifest.json"
        if not manifest_file.exists():
            continue
        
        try:
            manifest = IndexManifest.load(manifest_file)
            manifest = _resolve_manifest_paths(manifest, manifest_file)
            if manifest.root_ref == root_ref:
                matching_indices.append((manifest.created_at, manifest))
        except Exception as e:
            logger.warning(f"Failed to load manifest from {index_dir}: {e}")
            continue
    
    if not matching_indices:
        raise IndexNotFoundError(f"No indices found for root_ref: {root_ref}")
    
    # Sort by created_at timestamp (descending) and pick the most recent
    matching_indices.sort(key=lambda x: x[0], reverse=True)
    latest_manifest = matching_indices[0][1]
    
    logger.info(f"Resolved index by root_ref {root_ref}: {latest_manifest.index_id}")
    return latest_manifest
