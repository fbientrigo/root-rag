"""Index module: schema and index building."""
from root_rag.index.builder import build_full_index, build_index
from root_rag.index.fts import check_fts5_available
from root_rag.index.schemas import Chunk, IndexManifest

__all__ = [
    "Chunk",
    "IndexManifest",
    "build_index",
    "build_full_index",
    "check_fts5_available",
]
