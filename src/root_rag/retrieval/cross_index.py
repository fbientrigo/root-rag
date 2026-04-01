"""
Cross-index search: Query multiple indices simultaneously and merge results.

Enables queries like:
- "How does FairShip use TGeoManager?" (searches FairShip + ROOT indices)
- "TTree::Fill implementation and usage" (searches ROOT implementation + FairShip usage)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

from root_rag.index.locator import resolve_index
from root_rag.retrieval.lexical import lexical_search
from root_rag.retrieval.models import EvidenceCandidate

logger = logging.getLogger(__name__)


@dataclass
class IndexSource:
    """Configuration for a single index source."""
    name: str
    indexes_root: Path
    root_ref: Optional[str] = None
    index_id: Optional[str] = None
    weight: float = 1.0  # Weight for score normalization


class CrossIndexSearch:
    """Search across multiple indices and merge results."""
    
    def __init__(self, sources: List[IndexSource]):
        """Initialize cross-index search with multiple sources.
        
        Args:
            sources: List of IndexSource configurations
        """
        self.sources = sources
        self._resolved_indices = {}
        self._resolve_all_indices()
    
    def _resolve_all_indices(self):
        """Resolve all index sources to FTS databases."""
        for source in self.sources:
            try:
                manifest = resolve_index(
                    indexes_root=source.indexes_root,
                    root_ref=source.root_ref,
                    index_id=source.index_id,
                )
                
                # Build FTS DB path
                # Manifest.fts_db_path is stored as full path, extract just the filename
                fts_filename = Path(manifest.fts_db_path).name
                index_dir = source.indexes_root / manifest.index_id
                fts_db_path = index_dir / fts_filename
                
                if not fts_db_path.exists():
                    logger.warning(f"FTS database not found for {source.name}: {fts_db_path}")
                    continue
                
                self._resolved_indices[source.name] = {
                    "db_path": fts_db_path,
                    "manifest": manifest,
                    "weight": source.weight,
                }
                
                logger.info(
                    f"Resolved {source.name}: {manifest.chunk_count} chunks "
                    f"from {manifest.file_count} files (ref={manifest.root_ref})"
                )
                
            except Exception as e:
                logger.error(f"Failed to resolve index {source.name}: {e}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        per_index_limit: Optional[int] = None,
        query_mode: str = "baseline",
    ) -> List[EvidenceCandidate]:
        """Search all indices and merge results.
        
        Args:
            query: Search query string
            top_k: Total number of results to return
            per_index_limit: Max results per index (default: top_k)
            query_mode: Query transformation mode ('baseline' or 'lexnorm')
        
        Returns:
            Merged and re-ranked list of evidence candidates
        """
        if per_index_limit is None:
            per_index_limit = top_k
        
        # Collect results from all indices
        all_results = []
        
        for source_name, index_info in self._resolved_indices.items():
            try:
                results = lexical_search(
                    db_path=index_info["db_path"],
                    query=query,
                    top_k=per_index_limit,
                    query_mode=query_mode,
                )
                
                # Apply source weight to scores
                weight = index_info["weight"]
                for result in results:
                    result.score *= weight
                
                all_results.extend(results)
                logger.debug(f"Retrieved {len(results)} results from {source_name}")
                
            except Exception as e:
                logger.warning(f"Search failed for {source_name}: {e}")
        
        # Sort by score descending and return top_k
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(
            f"Cross-index search: {len(all_results)} total results "
            f"from {len(self._resolved_indices)} indices, returning top {top_k}"
        )
        
        return all_results[:top_k]
    
    def get_index_stats(self) -> Dict[str, Dict]:
        """Get statistics for all resolved indices.
        
        Returns:
            Dict mapping source names to index statistics
        """
        stats = {}
        for source_name, index_info in self._resolved_indices.items():
            manifest = index_info["manifest"]
            stats[source_name] = {
                "index_id": manifest.index_id,
                "root_ref": manifest.root_ref,
                "resolved_commit": manifest.resolved_commit,
                "chunk_count": manifest.chunk_count,
                "file_count": manifest.file_count,
                "weight": index_info["weight"],
            }
        return stats


def create_standard_search(
    data_root: Path = Path("data"),
    include_tier1: bool = True,
    include_sofie: bool = False,
    include_fairship: bool = True,
) -> CrossIndexSearch:
    """Create a standard cross-index search with common sources.
    
    Args:
        data_root: Root data directory (default: "data")
        include_tier1: Include ROOT Tier 1 index
        include_sofie: Include SOFIE index
        include_fairship: Include FairShip index
    
    Returns:
        Configured CrossIndexSearch instance
    """
    sources = []
    
    if include_tier1:
        sources.append(IndexSource(
            name="ROOT Tier 1",
            indexes_root=data_root / "indexes_tier1",
            root_ref="v6-36-08",
            weight=1.0,
        ))
    
    if include_sofie:
        sources.append(IndexSource(
            name="SOFIE",
            indexes_root=data_root / "indexes_sofie",
            root_ref="v6-36-08",
            weight=1.0,
        ))
    
    if include_fairship:
        sources.append(IndexSource(
            name="FairShip",
            indexes_root=data_root / "indexes_fairship",
            root_ref="master",
            weight=1.0,
        ))
    
    return CrossIndexSearch(sources=sources)
