"""Lexical retrieval wrappers."""
from pathlib import Path
from typing import List

from root_rag.retrieval.backends import build_retrieval_backend
from root_rag.retrieval.models import EvidenceCandidate
from root_rag.retrieval.pipeline import RetrievalPipeline
from root_rag.retrieval.transformers import build_query_transformer


def lexical_search(
    db_path: Path,
    query: str,
    top_k: int = 10,
    query_mode: str = "baseline",
    backend_name: str = "lexical_fts5",
) -> List[EvidenceCandidate]:
    """Search SQLite FTS5 index for query and return ranked evidence.

    `query_mode` controls pre-search query transformation:
    - `baseline`: identity transform
    - `lexnorm`: ROOT/FairShip lexical normalization and alias expansion
    """
    pipeline = RetrievalPipeline(
        backend=build_retrieval_backend(
            backend_name,
            db_path=Path(db_path),
        ),
        query_transformer=build_query_transformer(query_mode),
    )
    return pipeline.search(query, top_k=top_k)
