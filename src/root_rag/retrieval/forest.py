"""Experimental heterogeneous retrieval forest."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from root_rag.retrieval.interfaces import BaseRetrievalBackend, OperationalMetrics
from root_rag.retrieval.models import EvidenceCandidate

logger = logging.getLogger(__name__)

@dataclass
class RetrievalForestBackend(BaseRetrievalBackend):
    """Fuses multiple retrieval profiles (chunk sizes) using RRF."""

    profiles: List[BaseRetrievalBackend]
    profile_names: List[str]
    fusion_method: str = "rrf"
    dedup_method: str = "line_overlap"
    tie_breaker: str = "stable"
    rrf_k: int = 60
    backend_id: str = "retrieval_forest"

    def search(self, query: str, top_k: int) -> List[EvidenceCandidate]:
        if not self.profiles:
            return []

        all_results: List[List[EvidenceCandidate]] = []
        for backend, name in zip(self.profiles, self.profile_names):
            results = backend.search(query, top_k)
            # Add provenance
            for rank, res in enumerate(results, 1):
                res.source_profile = name
                res.original_rank = rank
            all_results.append(results)
        
        # Fusion
        if self.fusion_method == "concat":
            fused = []
            for profile_results in all_results:
                fused.extend(profile_results)
            # Sort by original rank first, then by profile order
            fused.sort(key=lambda x: (x.original_rank or 999, self.profile_names.index(x.source_profile or "")))
        else:
            fused = self._fuse(all_results, query)
        
        # Deduplication
        if self.dedup_method == "line_overlap":
            fused = self._deduplicate(fused)
            
        return fused[:top_k]

    def _fuse(self, results_per_profile: List[List[EvidenceCandidate]], query: str) -> List[EvidenceCandidate]:
        if self.fusion_method != "rrf":
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
            
        by_anchor: Dict[str, EvidenceCandidate] = {}
        fused_scores: Dict[str, float] = {}
        
        # For multi-chunking, chunks from different profiles covering same area 
        # should probably be treated as distinct but RRF can favor areas.
        # Here we use chunk_id as before to maintain continuity, 
        # but we'll improve the tie-breaking.
        
        for profile_results in results_per_profile:
            for rank, row in enumerate(profile_results, start=1):
                anchor = row.chunk_id
                if anchor not in by_anchor:
                    by_anchor[anchor] = row
                fused_scores[anchor] = fused_scores.get(anchor, 0.0) + 1.0 / (self.rrf_k + rank)
        
        if self.tie_breaker == "enhanced":
            # Deterministic tie-breaking rule:
            # 1. exact query term appears in file path or symbol-like span;
            # 2. smaller line span;
            # 3. earlier original rank;
            # 4. stable file path ordering.
            # (Skipping exact query term in chunk text as it's not in EvidenceCandidate)
            
            query_terms = [t.lower() for t in query.split() if len(t) > 2]
            
            def tie_break_key(anchor_id):
                cand = by_anchor[anchor_id]
                score = fused_scores[anchor_id]
                
                # Rule 1: Exact term in file path or symbol
                term_in_path = 0
                for term in query_terms:
                    if term in cand.file_path.lower():
                        term_in_path = 1
                        break
                    if cand.symbol_path and term in cand.symbol_path.lower():
                        term_in_path = 1
                        break
                
                span = cand.end_line - cand.start_line
                
                return (
                    -score,          # Primary: RRF score descending
                    -term_in_path,   # Rule 1: Match in path/symbol (1 > 0)
                    span,            # Rule 2: Smaller span ascending
                    cand.original_rank or 999, # Rule 3: Original rank ascending
                    cand.file_path,  # Rule 4: Stable file path
                    anchor_id
                )
            
            ordered_anchors = sorted(fused_scores.keys(), key=tie_break_key)
        else:
            ordered_anchors = sorted(
                fused_scores.keys(),
                key=lambda a: (-fused_scores[a], by_anchor[a].file_path, by_anchor[a].start_line, a)
            )
        
        return [by_anchor[a] for a in ordered_anchors]

    def _deduplicate(self, candidates: List[EvidenceCandidate]) -> List[EvidenceCandidate]:
        """Deduplicate overlapping line ranges from the same file."""
        unique_results: List[EvidenceCandidate] = []
        for cand in candidates:
            is_redundant = False
            for existing in unique_results:
                if cand.file_path == existing.file_path:
                    # Check for overlap: max(starts) <= min(ends)
                    if max(cand.start_line, existing.start_line) <= min(cand.end_line, existing.end_line):
                        is_redundant = True
                        break
            if not is_redundant:
                unique_results.append(cand)
        return unique_results

    def operational_metrics(self) -> OperationalMetrics:
        metrics = {
            "profile_count": len(self.profiles),
            "profile_names": self.profile_names,
            "fusion_method": self.fusion_method,
            "dedup_method": self.dedup_method,
        }
        # Aggregate index sizes if available
        total_size = 0.0
        for p in self.profiles:
            p_metrics = p.operational_metrics()
            if p_metrics.index_size_bytes:
                total_size += p_metrics.index_size_bytes
        
        metrics["total_index_size_bytes"] = total_size
        return self.normalize_operational_metrics(metrics)
