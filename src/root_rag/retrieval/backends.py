"""Retrieval backends."""

from __future__ import annotations

import hashlib
import math
import logging
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from root_rag.retrieval.interfaces import BaseRetrievalBackend, OperationalMetrics
from root_rag.retrieval.models import EvidenceCandidate
from root_rag.retrieval.s1_semantic import (
    LocalEmbedder,
    SemanticFaissSearcher,
    SentenceTransformerLocalEmbedder,
    fuse_ranked_results,
    is_symbol_like_query,
)

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
IDENTIFIER_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")
SEMANTIC_ALIAS_MAP: Dict[str, tuple[str, ...]] = {
    "assembly": ("addnode", "volume", "construct"),
    "branch": ("tree", "address", "entry"),
    "declaration": ("class", "struct", "header", "interface"),
    "detector": ("hit", "module", "geometry"),
    "geometry": ("volume", "shape", "node"),
    "implementation": ("override", "source", "definition"),
    "loader": ("reader", "branch", "file"),
    "navigation": ("manager", "volume", "node"),
    "pattern": ("usage", "example", "flow"),
    "storage": ("stack", "array", "buffer"),
}


def _sanitize_fts_query(query: str) -> str:
    """Sanitize query string for FTS5 MATCH operator."""
    query = re.sub(r"\b(where|what|how|is|are|was|were|the|a|an)\b", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"[?!.,;]", " ", query)

    words = query.split()
    quoted_words = []

    for word in words:
        word = word.strip()
        if not word:
            continue

        if word.upper() in ("AND", "OR", "NOT", "NEAR"):
            quoted_words.append(word)
            continue

        if re.search(r'[:\(\)\*\-"]', word):
            word = '"' + word.replace('"', '""') + '"'

        quoted_words.append(word)

    if not quoted_words:
        return '""'
    return " ".join(quoted_words)


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _split_identifier_parts(token: str) -> List[str]:
    normalized = token.replace("::", "_")
    parts = IDENTIFIER_PART_RE.findall(normalized)
    return [part.lower() for part in parts if part]


def _expand_semantic_features(text: str) -> List[str]:
    features: List[str] = []
    for token in _tokenize(text):
        features.append(token)

        for part in _split_identifier_parts(token):
            if part != token:
                features.append(f"part:{part}")

        if len(token) >= 5:
            for idx in range(len(token) - 2):
                features.append(f"tri:{token[idx:idx + 3]}")

        for alias in SEMANTIC_ALIAS_MAP.get(token, ()):
            features.append(f"alias:{alias}")

    return features


def _stable_u64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _build_hashed_dense_vector(tokens: List[str], dim: int) -> List[float]:
    vector = [0.0] * dim
    for token in tokens:
        idx = _stable_u64(token) % dim
        sign = 1.0 if (_stable_u64(f"sign:{token}") & 1) == 0 else -1.0
        vector[idx] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0.0:
        inv_norm = 1.0 / norm
        vector = [value * inv_norm for value in vector]
    return vector


def _dot_product(a: List[float], b: List[float]) -> float:
    return sum(left * right for left, right in zip(a, b))


def _nonzero_count(vector: List[float]) -> int:
    return sum(1 for value in vector if value != 0.0)


def _infer_language_from_path(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}:
        return "cpp"
    return "txt"


@dataclass
class FTS5LexicalBackend(BaseRetrievalBackend):
    """SQLite FTS5 backend implementation."""

    db_path: Path
    backend_id: str = "lexical_fts5"

    def search(self, query: str, top_k: int) -> List[EvidenceCandidate]:
        top_k = self.normalize_top_k(top_k)
        if top_k == 0:
            return []

        db_path = Path(self.db_path)
        if not db_path.exists():
            logger.warning(f"Database does not exist: {db_path}")
            return []

        try:
            with sqlite3.connect(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                fts_query = _sanitize_fts_query(query)
                cursor.execute(
                    """
                    SELECT
                        chunk_id,
                        file_path,
                        start_line,
                        end_line,
                        symbol_path,
                        doc_origin,
                        language,
                        root_ref,
                        resolved_commit,
                        bm25(chunks_fts) AS score
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    ORDER BY score, file_path, start_line
                    LIMIT ?
                    """,
                    (fts_query, top_k),
                )

                results = []
                for row in cursor.fetchall():
                    results.append(
                        EvidenceCandidate(
                            chunk_id=row["chunk_id"],
                            file_path=row["file_path"],
                            start_line=row["start_line"],
                            end_line=row["end_line"],
                            symbol_path=row["symbol_path"] if row["symbol_path"] else None,
                            doc_origin=row["doc_origin"],
                            language=row["language"],
                            root_ref=row["root_ref"],
                            resolved_commit=row["resolved_commit"],
                            score=row["score"],
                        )
                    )

            logger.info(f"Lexical search for '{query}': {len(results)} results (top_k={top_k})")
            return results

        except sqlite3.Error as exc:
            logger.error(f"FTS5 search failed: {exc}")
            return []

    def operational_metrics(self) -> OperationalMetrics:
        db_path = Path(self.db_path)
        size_bytes = float(db_path.stat().st_size) if db_path.exists() else None
        return self.normalize_operational_metrics(
            {
                "index_size_bytes": size_bytes,
            }
        )


@dataclass
class InMemoryBM25LexicalBackend(BaseRetrievalBackend):
    """Deterministic BM25 backend over corpus JSONL rows."""

    corpus_rows: List[dict]
    k1: float = 1.5
    b: float = 0.75
    corpus_artifact_path: Optional[Path] = None
    backend_id: str = "lexical_bm25_memory"
    _doc_chunks: List[str] = field(default_factory=list, init=False, repr=False)
    _doc_tfs: List[Counter] = field(default_factory=list, init=False, repr=False)
    _doc_lens: List[int] = field(default_factory=list, init=False, repr=False)
    _df: Counter = field(default_factory=Counter, init=False, repr=False)
    _doc_count: int = field(default=0, init=False, repr=False)
    _avgdl: float = field(default=0.0, init=False, repr=False)
    _row_by_chunk: Dict[str, dict] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for row in self.corpus_rows:
            doc_text = (
                f"{row['text']} "
                f"{row['file_path']} "
                f"{' '.join(row.get('headers_used', []))}"
            )
            tokens = _tokenize(doc_text)
            tf = Counter(tokens)
            chunk_id = row["chunk_id"]
            self._doc_chunks.append(chunk_id)
            self._doc_tfs.append(tf)
            self._doc_lens.append(len(tokens))
            self._row_by_chunk[chunk_id] = row
            for token in tf.keys():
                self._df[token] += 1

        self._doc_count = len(self._doc_chunks)
        self._avgdl = (sum(self._doc_lens) / self._doc_count) if self._doc_count else 0.0

    def _idf(self, token: str) -> float:
        n = self._df.get(token, 0)
        if n == 0:
            return 0.0
        return math.log(((self._doc_count - n + 0.5) / (n + 0.5)) + 1.0)

    def search(self, query: str, top_k: int) -> List[EvidenceCandidate]:
        top_k = self.normalize_top_k(top_k)
        if top_k == 0 or self._doc_count == 0 or self._avgdl <= 0.0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for idx, chunk_id in enumerate(self._doc_chunks):
            tf = self._doc_tfs[idx]
            dl = self._doc_lens[idx]
            total = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if freq == 0:
                    continue
                idf = self._idf(token)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl))
                total += idf * ((freq * (self.k1 + 1.0)) / denom)

            if total > 0.0:
                scored.append((chunk_id, total))

        scored.sort(key=lambda row: (-row[1], row[0]))
        scored = scored[:top_k]

        results = []
        for chunk_id, score in scored:
            row = self._row_by_chunk[chunk_id]
            line_range = row.get("line_range", [1, 1])
            start_line = int(line_range[0]) if line_range else 1
            end_line = int(line_range[1]) if len(line_range) >= 2 else start_line
            file_path = row.get("file_path", "")
            results.append(
                EvidenceCandidate(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    symbol_path=None,
                    doc_origin=row.get("provenance", "benchmark_artifact"),
                    language=row.get("language", _infer_language_from_path(file_path)),
                    root_ref=row.get("root_ref", "benchmark_artifact"),
                    resolved_commit=row.get("resolved_commit", "benchmark_artifact"),
                    score=score,
                )
            )
        return results

    def operational_metrics(self) -> OperationalMetrics:
        metrics: OperationalMetrics = {
            "k1": self.k1,
            "b": self.b,
            "avgdl": self._avgdl,
            "docs": float(self._doc_count),
            "index_size_bytes": None,
        }
        if self.corpus_artifact_path:
            corpus_path = Path(self.corpus_artifact_path)
            if corpus_path.exists():
                metrics["corpus_size_bytes"] = float(corpus_path.stat().st_size)
        return self.normalize_operational_metrics(metrics)


@dataclass
class DenseHashMemoryBackend(BaseRetrievalBackend):
    """Deterministic dense-hash backend over corpus JSONL rows.

    This is an opt-in dense retrieval scaffold with no external embedding model dependency.
    """

    corpus_rows: List[dict]
    vector_dim: int = 256
    corpus_artifact_path: Optional[Path] = None
    backend_id: str = "dense_hash_memory"
    _doc_chunks: List[str] = field(default_factory=list, init=False, repr=False)
    _doc_vectors: List[List[float]] = field(default_factory=list, init=False, repr=False)
    _row_by_chunk: Dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _doc_count: int = field(default=0, init=False, repr=False)
    _avg_nonzero_dims: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.vector_dim = int(self.vector_dim)
        if self.vector_dim <= 0:
            raise ValueError("vector_dim must be a positive integer")

        nonzero_total = 0
        for row in self.corpus_rows:
            doc_text = (
                f"{row['text']} "
                f"{row['file_path']} "
                f"{' '.join(row.get('headers_used', []))}"
            )
            tokens = _tokenize(doc_text)
            chunk_id = row["chunk_id"]
            vector = _build_hashed_dense_vector(tokens, self.vector_dim)
            self._doc_chunks.append(chunk_id)
            self._doc_vectors.append(vector)
            self._row_by_chunk[chunk_id] = row
            nonzero_total += _nonzero_count(vector)

        self._doc_count = len(self._doc_chunks)
        self._avg_nonzero_dims = (nonzero_total / self._doc_count) if self._doc_count else 0.0

    def search(self, query: str, top_k: int) -> List[EvidenceCandidate]:
        top_k = self.normalize_top_k(top_k)
        if top_k == 0 or self._doc_count == 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_vector = _build_hashed_dense_vector(query_tokens, self.vector_dim)
        if _nonzero_count(query_vector) == 0:
            return []

        scored = []
        for idx, chunk_id in enumerate(self._doc_chunks):
            score = _dot_product(query_vector, self._doc_vectors[idx])
            if score > 0.0:
                scored.append((chunk_id, score))

        scored.sort(key=lambda row: (-row[1], row[0]))
        scored = scored[:top_k]

        results = []
        for chunk_id, score in scored:
            row = self._row_by_chunk[chunk_id]
            line_range = row.get("line_range", [1, 1])
            start_line = int(line_range[0]) if line_range else 1
            end_line = int(line_range[1]) if len(line_range) >= 2 else start_line
            file_path = row.get("file_path", "")
            results.append(
                EvidenceCandidate(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    symbol_path=None,
                    doc_origin=row.get("provenance", "benchmark_artifact"),
                    language=row.get("language", _infer_language_from_path(file_path)),
                    root_ref=row.get("root_ref", "benchmark_artifact"),
                    resolved_commit=row.get("resolved_commit", "benchmark_artifact"),
                    score=score,
                )
            )
        return results

    def operational_metrics(self) -> OperationalMetrics:
        metrics: OperationalMetrics = {
            "vector_dim": self.vector_dim,
            "similarity": "cosine_hash",
            "docs": float(self._doc_count),
            "avg_nonzero_dims": self._avg_nonzero_dims,
            "index_size_bytes": None,
        }
        if self.corpus_artifact_path:
            corpus_path = Path(self.corpus_artifact_path)
            if corpus_path.exists():
                metrics["corpus_size_bytes"] = float(corpus_path.stat().st_size)
        return self.normalize_operational_metrics(metrics)


@dataclass
class SemanticHashMemoryBackend(BaseRetrievalBackend):
    """Deterministic semantic-style backend over corpus JSONL rows.

    This backend is local-only and approximates semantic similarity with
    expanded identifier parts, alias features, and character trigrams.
    """

    corpus_rows: List[dict]
    vector_dim: int = 512
    corpus_artifact_path: Optional[Path] = None
    backend_id: str = "semantic_hash_memory"
    _doc_chunks: List[str] = field(default_factory=list, init=False, repr=False)
    _doc_vectors: List[List[float]] = field(default_factory=list, init=False, repr=False)
    _row_by_chunk: Dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _doc_count: int = field(default=0, init=False, repr=False)
    _avg_nonzero_dims: float = field(default=0.0, init=False, repr=False)
    _avg_feature_tokens: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.vector_dim = int(self.vector_dim)
        if self.vector_dim <= 0:
            raise ValueError("vector_dim must be a positive integer")

        nonzero_total = 0
        feature_total = 0
        for row in self.corpus_rows:
            doc_text = (
                f"{row['text']} "
                f"{row['file_path']} "
                f"{' '.join(row.get('headers_used', []))}"
            )
            features = _expand_semantic_features(doc_text)
            chunk_id = row["chunk_id"]
            vector = _build_hashed_dense_vector(features, self.vector_dim)
            self._doc_chunks.append(chunk_id)
            self._doc_vectors.append(vector)
            self._row_by_chunk[chunk_id] = row
            nonzero_total += _nonzero_count(vector)
            feature_total += len(features)

        self._doc_count = len(self._doc_chunks)
        if self._doc_count:
            self._avg_nonzero_dims = nonzero_total / self._doc_count
            self._avg_feature_tokens = feature_total / self._doc_count

    def search(self, query: str, top_k: int) -> List[EvidenceCandidate]:
        top_k = self.normalize_top_k(top_k)
        if top_k == 0 or self._doc_count == 0:
            return []

        query_features = _expand_semantic_features(query)
        if not query_features:
            return []

        query_vector = _build_hashed_dense_vector(query_features, self.vector_dim)
        if _nonzero_count(query_vector) == 0:
            return []

        scored = []
        for idx, chunk_id in enumerate(self._doc_chunks):
            score = _dot_product(query_vector, self._doc_vectors[idx])
            if score > 0.0:
                scored.append((chunk_id, score))

        scored.sort(key=lambda row: (-row[1], row[0]))
        scored = scored[:top_k]

        results = []
        for chunk_id, score in scored:
            row = self._row_by_chunk[chunk_id]
            line_range = row.get("line_range", [1, 1])
            start_line = int(line_range[0]) if line_range else 1
            end_line = int(line_range[1]) if len(line_range) >= 2 else start_line
            file_path = row.get("file_path", "")
            results.append(
                EvidenceCandidate(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    symbol_path=None,
                    doc_origin=row.get("provenance", "benchmark_artifact"),
                    language=row.get("language", _infer_language_from_path(file_path)),
                    root_ref=row.get("root_ref", "benchmark_artifact"),
                    resolved_commit=row.get("resolved_commit", "benchmark_artifact"),
                    score=score,
                )
            )
        return results

    def operational_metrics(self) -> OperationalMetrics:
        metrics: OperationalMetrics = {
            "vector_dim": self.vector_dim,
            "similarity": "cosine_semantic_hash",
            "docs": float(self._doc_count),
            "avg_nonzero_dims": self._avg_nonzero_dims,
            "avg_feature_tokens": self._avg_feature_tokens,
            "index_size_bytes": None,
        }
        if self.corpus_artifact_path:
            corpus_path = Path(self.corpus_artifact_path)
            if corpus_path.exists():
                metrics["corpus_size_bytes"] = float(corpus_path.stat().st_size)
        return self.normalize_operational_metrics(metrics)


@dataclass
class SemanticFaissBackend(BaseRetrievalBackend):
    """S1 local-embedding semantic backend over persisted FAISS exact artifacts."""

    semantic_manifest_path: Path
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 16
    local_files_only: bool = True
    embedder: Optional[LocalEmbedder] = None
    searcher: Optional[SemanticFaissSearcher] = None
    backend_id: str = "semantic_faiss"

    def __post_init__(self) -> None:
        if self.embedder is None:
            self.embedder = SentenceTransformerLocalEmbedder(
                model_name=self.model_name,
                device=self.device,
                batch_size=self.batch_size,
                local_files_only=self.local_files_only,
            )
        if self.searcher is None:
            self.searcher = SemanticFaissSearcher(
                manifest_path=Path(self.semantic_manifest_path),
                embedder=self.embedder,
            )

    def search(self, query: str, top_k: int) -> List[EvidenceCandidate]:
        top_k = self.normalize_top_k(top_k)
        if top_k == 0:
            return []
        assert self.searcher is not None
        return self.searcher.search(query, top_k=top_k)

    def operational_metrics(self) -> OperationalMetrics:
        assert self.searcher is not None
        manifest = self.searcher.manifest
        index_path = Path(manifest.index_path)
        metrics: OperationalMetrics = {
            "vector_dim": manifest.embedding_dimension,
            "similarity": "cosine_faiss_ip",
            "docs": float(manifest.row_count),
            "model_name": manifest.model_name,
            "normalization": manifest.normalization,
            "faiss_index_type": manifest.faiss_index_type,
            "index_size_bytes": float(index_path.stat().st_size) if index_path.exists() else None,
            "semantic_manifest_path": str(self.semantic_manifest_path),
        }
        return self.normalize_operational_metrics(metrics)


@dataclass
class HybridS1Backend(BaseRetrievalBackend):
    """S1 hybrid backend: lexical backbone plus opt-in semantic FAISS fusion."""

    lexical_backend: BaseRetrievalBackend
    semantic_backend: BaseRetrievalBackend
    lexical_weight: float = 0.45
    semantic_weight: float = 0.55
    lexical_weight_symbol: float = 0.85
    semantic_weight_symbol: float = 0.15
    lexical_pin_count: int = 3
    search_depth: int = 50
    backend_id: str = "hybrid_s1"

    def search(self, query: str, top_k: int) -> List[EvidenceCandidate]:
        top_k = self.normalize_top_k(top_k)
        if top_k == 0:
            return []

        search_depth = max(top_k, self.lexical_pin_count, self.search_depth)
        lexical_results = self.lexical_backend.search(query, top_k=search_depth)
        semantic_results = self.semantic_backend.search(query, top_k=search_depth)
        symbol_like = is_symbol_like_query(query)

        if symbol_like:
            lexical_weight = self.lexical_weight_symbol
            semantic_weight = self.semantic_weight_symbol
        else:
            lexical_weight = self.lexical_weight
            semantic_weight = self.semantic_weight

        return fuse_ranked_results(
            lexical_results=lexical_results,
            semantic_results=semantic_results,
            top_k=top_k,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
            symbol_safe=symbol_like,
            lexical_pin_count=self.lexical_pin_count,
        )

    def operational_metrics(self) -> OperationalMetrics:
        lexical_metrics = self.lexical_backend.operational_metrics()
        semantic_metrics = self.semantic_backend.operational_metrics()
        metrics: OperationalMetrics = {
            "fusion": "weighted_rrf",
            "lexical_backend": getattr(self.lexical_backend, "backend_id", "unknown"),
            "semantic_backend": getattr(self.semantic_backend, "backend_id", "unknown"),
            "lexical_weight": self.lexical_weight,
            "semantic_weight": self.semantic_weight,
            "lexical_weight_symbol": self.lexical_weight_symbol,
            "semantic_weight_symbol": self.semantic_weight_symbol,
            "lexical_pin_count": self.lexical_pin_count,
            "search_depth": self.search_depth,
            "docs": semantic_metrics.get("docs"),
            "vector_dim": semantic_metrics.get("vector_dim"),
            "model_name": semantic_metrics.get("model_name"),
            "semantic_manifest_path": semantic_metrics.get("semantic_manifest_path"),
        }
        if lexical_metrics.get("index_size_bytes") is not None:
            metrics["lexical_index_size_bytes"] = lexical_metrics.get("index_size_bytes")
        if semantic_metrics.get("index_size_bytes") is not None:
            metrics["semantic_index_size_bytes"] = semantic_metrics.get("index_size_bytes")
        return self.normalize_operational_metrics(metrics)


def build_retrieval_backend(
    name: str,
    *,
    db_path: Optional[Path] = None,
    corpus_rows: Optional[List[dict]] = None,
    corpus_artifact_path: Optional[Path] = None,
    semantic_manifest_path: Optional[Path] = None,
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    semantic_device: str = "cpu",
    semantic_batch_size: int = 16,
    semantic_local_files_only: bool = True,
    k1: float = 1.5,
    b: float = 0.75,
    dense_dim: int = 256,
) -> BaseRetrievalBackend:
    """Factory for retrieval backends used by runtime and benchmarks."""
    normalized = name.strip().lower()

    if normalized in {"lexical_fts5", "fts5", "lexical"}:
        if db_path is None:
            raise ValueError("db_path is required for lexical_fts5 backend")
        return FTS5LexicalBackend(Path(db_path))

    if normalized in {"lexical_bm25_memory", "bm25_memory", "bm25"}:
        if corpus_rows is None:
            raise ValueError("corpus_rows are required for lexical_bm25_memory backend")
        return InMemoryBM25LexicalBackend(
            corpus_rows=corpus_rows,
            k1=k1,
            b=b,
            corpus_artifact_path=corpus_artifact_path,
        )

    if normalized in {"dense_hash_memory", "dense_hash", "dense"}:
        if corpus_rows is None:
            raise ValueError("corpus_rows are required for dense_hash_memory backend")
        return DenseHashMemoryBackend(
            corpus_rows=corpus_rows,
            vector_dim=dense_dim,
            corpus_artifact_path=corpus_artifact_path,
        )

    if normalized in {"semantic_hash_memory", "semantic_hash", "semantic"}:
        if corpus_rows is None:
            raise ValueError("corpus_rows are required for semantic_hash_memory backend")
        return SemanticHashMemoryBackend(
            corpus_rows=corpus_rows,
            vector_dim=dense_dim,
            corpus_artifact_path=corpus_artifact_path,
        )

    if normalized in {"semantic_faiss", "semantic_s1", "s1_semantic"}:
        if semantic_manifest_path is None:
            raise ValueError("semantic_manifest_path is required for semantic_faiss backend")
        return SemanticFaissBackend(
            semantic_manifest_path=Path(semantic_manifest_path),
            model_name=semantic_model_name,
            device=semantic_device,
            batch_size=semantic_batch_size,
            local_files_only=semantic_local_files_only,
        )

    if normalized in {"hybrid_s1", "s1", "hybrid"}:
        if semantic_manifest_path is None:
            raise ValueError("semantic_manifest_path is required for hybrid_s1 backend")
        if db_path is not None:
            lexical_backend: BaseRetrievalBackend = FTS5LexicalBackend(Path(db_path))
        elif corpus_rows is not None:
            lexical_backend = InMemoryBM25LexicalBackend(
                corpus_rows=corpus_rows,
                k1=k1,
                b=b,
                corpus_artifact_path=corpus_artifact_path,
            )
        else:
            raise ValueError("db_path or corpus_rows are required for hybrid_s1 backend")
        semantic_backend = SemanticFaissBackend(
            semantic_manifest_path=Path(semantic_manifest_path),
            model_name=semantic_model_name,
            device=semantic_device,
            batch_size=semantic_batch_size,
            local_files_only=semantic_local_files_only,
        )
        return HybridS1Backend(
            lexical_backend=lexical_backend,
            semantic_backend=semantic_backend,
        )

    raise ValueError(f"Unknown retrieval backend: {name}")
