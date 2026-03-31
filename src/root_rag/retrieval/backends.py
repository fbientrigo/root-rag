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

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


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


def build_retrieval_backend(
    name: str,
    *,
    db_path: Optional[Path] = None,
    corpus_rows: Optional[List[dict]] = None,
    corpus_artifact_path: Optional[Path] = None,
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

    raise ValueError(f"Unknown retrieval backend: {name}")
