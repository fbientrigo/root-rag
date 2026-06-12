"""Root RAG CLI - Command line interface for root-rag."""
import json as json_module
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click

from root_rag.corpus import fetch_corpus, InvalidRefError
from root_rag.core.errors import IndexNotFoundError
from root_rag.index import build_full_index, check_fts5_available
from root_rag.index.locator import resolve_index
from root_rag.index.schemas import IndexManifest
from root_rag.retrieval import build_retrieval_backend, lexical_search
from root_rag.retrieval.models import EvidenceCandidate
from root_rag.retrieval.s1_semantic import (
    SentenceTransformerLocalEmbedder,
    _slugify_model_name,
    build_semantic_index_artifacts,
    load_corpus_rows,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
)
logger = logging.getLogger("root_rag")


INDEX_PROFILES: Dict[str, Path] = {
    "root": Path("data/indexes"),
    "fairship": Path("data/indexes_fairship"),
    "project_docs": Path("data/indexes_project_docs"),
}


def _select_indexes_root(index_dir: Optional[Path], profile: str, index_id: Optional[str]) -> Path:
    """Resolve index root with optional profile and fallback for explicit index IDs."""
    if index_dir is not None:
        return Path(index_dir)

    if profile in INDEX_PROFILES:
        return INDEX_PROFILES[profile]

    if index_id:
        for candidate in INDEX_PROFILES.values():
            if (candidate / index_id / "index_manifest.json").exists():
                return candidate

    return INDEX_PROFILES["root"]


def _parse_file_range(raw: str) -> Tuple[str, int, int]:
    match = re.fullmatch(r"(.+):(\d+)-(\d+)", raw.strip())
    if match is None:
        raise click.UsageError("Range must be formatted as <file:start-end>, e.g. shipgen/MuDISGenerator.cxx:71-150")

    file_path = match.group(1)
    start_line = int(match.group(2))
    end_line = int(match.group(3))
    if start_line < 1 or end_line < 1 or end_line < start_line:
        raise click.UsageError("Invalid line range: start and end must be positive and end >= start")
    return file_path, start_line, end_line


def _load_indexed_file_lines(chunks_path: Path, file_path: str) -> Tuple[bool, Dict[int, str]]:
    """Reconstruct line text from indexed chunk content only (no source checkout dependency)."""
    found_file = False
    line_map: Dict[int, str] = {}

    with open(chunks_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            row = json_module.loads(raw)
            if row.get("file_path") != file_path:
                continue

            found_file = True
            start_line = int(row["start_line"])
            end_line = int(row["end_line"])
            content_lines = row.get("content", "").splitlines()

            max_count = max(0, end_line - start_line + 1)
            for offset, text in enumerate(content_lines[:max_count]):
                line_no = start_line + offset
                if line_no not in line_map:
                    line_map[line_no] = text

    return found_file, line_map


def _resolve_chunks_path(manifest: IndexManifest, indexes_root: Path) -> Path:
    """Resolve chunks path with compatibility fallback for older relative manifests."""
    chunks_path = Path(manifest.chunks_path)
    if chunks_path.exists():
        return chunks_path

    manifest_file = Path(indexes_root) / manifest.index_id / "index_manifest.json"
    if manifest_file.exists():
        raw_manifest = IndexManifest.load(manifest_file)
        raw_chunks_path = Path(raw_manifest.chunks_path)
        if raw_chunks_path.is_absolute():
            if raw_chunks_path.exists():
                return raw_chunks_path
        else:
            cwd_candidate = raw_chunks_path.resolve()
            if cwd_candidate.exists():
                return cwd_candidate
            data_candidate = (Path("data") / raw_chunks_path).resolve()
            if data_candidate.exists():
                return data_candidate
            manifest_dir_candidate = (manifest_file.parent / raw_chunks_path).resolve()
            if manifest_dir_candidate.exists():
                return manifest_dir_candidate

    return chunks_path


def _resolve_semantic_manifest_path(index_manifest: IndexManifest, explicit_path: Optional[Path]) -> Optional[Path]:
    if explicit_path:
        return explicit_path
    if index_manifest.semantic_manifest_path:
        return Path(index_manifest.semantic_manifest_path)
    return None


def _resolve_backend_and_results(
    query: str,
    manifest: IndexManifest,
    top_k: int,
    retrieval_backend: str,
    retrieval_forest: Optional[str],
    fusion: str,
    dedup: str,
    tie_breaker: str,
    baseline: bool,
    profile: str = "root",
    semantic_manifest: Optional[Path] = None,
    semantic_model: str = "",
) -> Tuple[List[EvidenceCandidate], str, Optional[str]]:
    """Internal helper to resolve backend and execute search with fallbacks.

    Returns (results, actual_backend_name, fallback_reason).
    """
    requested_backend = retrieval_backend.strip().lower()
    if baseline:
        requested_backend = "lexical"

    # FairShip-only forest default logic
    is_auto_forest = False
    if requested_backend == "lexical" and not baseline and profile.lower() == "fairship":
        # Check if forest config exists
        config_path = Path("configs/retrieval_forest_profiles.json")
        if config_path.exists():
            is_auto_forest = True
            requested_backend = "forest"

    db_path = manifest.fts_db_path
    fallback_reason = None

    if requested_backend == "forest":
        config_path = Path("configs/retrieval_forest_profiles.json")
        if not config_path.exists():
            if not is_auto_forest:
                raise click.UsageError(f"Forest config not found: {config_path}")
            fallback_reason = "configs/retrieval_forest_profiles.json missing"
            requested_backend = "lexical"
        else:
            with open(config_path, "r") as f:
                config = json_module.load(f)

            forest_profiles = retrieval_forest or ",".join(config.get("default_profiles", []))
            if not forest_profiles:
                if not is_auto_forest:
                    raise click.UsageError("No forest profiles specified or found in config")
                fallback_reason = "no default profiles in forest config"
                requested_backend = "lexical"
            else:
                requested_profile_list = [p.strip() for p in forest_profiles.split(",")]
                forest_db_paths = []
                forest_profile_names = []

                missing_in_config = []
                commit_mismatch = []
                missing_on_disk = []

                for p_name in requested_profile_list:
                    found_in_config = False
                    for p_entry in config["profiles"]:
                        if p_entry["profile_id"] == p_name:
                            found_in_config = True
                            # Verify commit matches to avoid cross-version contamination
                            target_commit = manifest.resolved_commit[:12]
                            entry_commit = (p_entry.get("source_commit") or "")[:12]
                            if entry_commit and entry_commit != target_commit:
                                commit_mismatch.append(f"{p_name} (needs {entry_commit}, got {target_commit})")
                                continue

                            db_path_entry = Path(p_entry["index_output_path"]) / "fts.sqlite"
                            if db_path_entry.exists():
                                forest_db_paths.append(db_path_entry)
                                forest_profile_names.append(p_name)
                            else:
                                missing_on_disk.append(str(db_path_entry))
                            break
                    if not found_in_config:
                        missing_in_config.append(p_name)

                # Decision: must have all requested profiles to use forest
                error_msgs = []
                if missing_in_config:
                    error_msgs.append(f"profiles not in config: {', '.join(missing_in_config)}")
                if commit_mismatch:
                    error_msgs.append(f"commit mismatch: {', '.join(commit_mismatch)}")
                if missing_on_disk:
                    error_msgs.append(f"indexes missing on disk: {', '.join(missing_on_disk)}")

                if not error_msgs and forest_db_paths:
                    backend = build_retrieval_backend(
                        "retrieval_forest",
                        forest_db_paths=forest_db_paths,
                        forest_profile_names=forest_profile_names,
                        fusion_method=fusion,
                        dedup_method=dedup,
                        tie_breaker=tie_breaker,
                    )
                    return backend.search(query, top_k=top_k), "forest", None
                else:
                    fallback_reason = "; ".join(error_msgs)
                    if not is_auto_forest:
                        raise click.UsageError(f"Forest backend requested but unavailable: {fallback_reason}")
                    requested_backend = "lexical"

    if requested_backend == "lexical":
        return lexical_search(
            db_path=str(db_path),
            query=query,
            top_k=top_k,
        ), "lexical", fallback_reason

    # Semantic / Hybrid
    semantic_manifest_path = _resolve_semantic_manifest_path(manifest, semantic_manifest)
    if semantic_manifest_path is None or not semantic_manifest_path.exists():
        if requested_backend in {"semantic", "hybrid"}:
            raise click.UsageError("S1 semantic manifest not found. Build semantic artifacts first or pass --semantic-manifest.")
        return lexical_search(db_path=str(db_path), query=query, top_k=top_k), "lexical (fallback)", fallback_reason

    actual_name = "semantic" if requested_backend == "semantic" else "hybrid"
    backend = build_retrieval_backend(
        "semantic_faiss" if requested_backend == "semantic" else "hybrid_s1",
        db_path=Path(db_path),
        semantic_manifest_path=semantic_manifest_path,
        semantic_model_name=semantic_model,
    )
    return backend.search(query, top_k=top_k), actual_name, fallback_reason


@click.group()
def main():
    """Root RAG - A hybrid retrieval-based RAG system."""
    pass


@main.command()
@click.option(
    "--root-ref",
    required=True,
    help="Branch, tag, or commit to fetch",
)
@click.option(
    "--repo-url",
    default="https://github.com/root-project/root.git",
    help="Repository URL (default: ROOT official repository)",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/raw/corpora"),
    help="Cache directory for corpora",
)
@click.option(
    "--force-refresh",
    is_flag=True,
    help="Ignore cache and fetch fresh",
)
def fetch(root_ref: str, repo_url: str, cache_dir: Path, force_refresh: bool):
    """Fetch a ROOT corpus revision and write manifest.
    
    Examples:
        root-rag fetch --root-ref v6-32-00
        root-rag fetch --root-ref master --cache-dir ~/.cache/root-rag
    
    Exit codes:
        0: Success
        1: Generic runtime failure
        2: Invalid CLI usage
        3: Requested revision not found
    """
    try:
        manifest = fetch_corpus(
            repo_url=repo_url,
            root_ref=root_ref,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )
        
        click.echo(f"[OK] Corpus fetched: {root_ref}")
        click.echo(f"  Commit: {manifest.resolved_commit[:12]}")
        click.echo(f"  Path: {manifest.local_path}")
        click.echo(f"  Manifest: {cache_dir / f'{root_ref}__{manifest.resolved_commit[:12]}' / 'manifest.json'}")
        sys.exit(0)
    
    except InvalidRefError as e:
        logger.error(f"Invalid reference: {str(e)}")
        sys.exit(3)
    
    except Exception as e:
        logger.error(f"Fetch failed: {str(e)}")
        sys.exit(1)


@main.command()
@click.option(
    "--root-ref",
    default="v6-36-08",
    help="Branch, tag, or commit to index (default: v6-36-08 from FairShip)",
)
@click.option(
    "--repo-url",
    default="https://github.com/root-project/root.git",
    help="Repository URL (default: ROOT official repository)",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/raw/corpora"),
    help="Cache directory for corpora",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/indexes"),
    help="Output directory for indexes",
)
@click.option(
    "--window-lines",
    type=int,
    default=80,
    help="Lines per chunk window",
)
@click.option(
    "--overlap-lines",
    type=int,
    default=10,
    help="Overlap between windows",
)
@click.option(
    "--seed-corpus",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Seed corpus config (auto-detects for v6-36-08)",
)
@click.option(
    "--full-corpus",
    is_flag=True,
    help="Index full corpus (ignore seed config)",
)
def index(
    root_ref: str,
    repo_url: str,
    cache_dir: Path,
    output_dir: Path,
    window_lines: int,
    overlap_lines: int,
    seed_corpus: Path,
    full_corpus: bool,
):
    """Build FTS5 lexical search index for ROOT corpus.
    
    Creates SQLite FTS5 index for fast full-text search over ROOT code.
    Automatically fetches corpus if not cached.
    
    For ROOT 6.36.08 (FairShip anchor), uses seed corpus by default.
    
    Examples:
        root-rag index                    # Index v6-36-08 with seed corpus
        root-rag index --full-corpus      # Index v6-36-08 fully
        root-rag index --root-ref master  # Index different version
    
    Exit codes:
        0: Success
        1: Generic runtime failure
        2: Invalid CLI arguments
        3: Requested revision not found
        7: Configuration file error
        8: FTS5 not available on this system
    """
    
    # Auto-detect seed corpus config for v6-36-08
    if not full_corpus and not seed_corpus:
        if root_ref == "v6-36-08":
            default_seed = Path("configs/seed_corpus_root_636.yaml")
            if default_seed.exists():
                seed_corpus = default_seed
                logger.info(f"Using seed corpus config: {seed_corpus}")
                click.echo(f"[INFO] Using FairShip-focused seed corpus for {root_ref}")
                click.echo(f"       Config: {seed_corpus}")
                click.echo(f"       (Use --full-corpus to index all files)")
            else:
                logger.warning(f"Seed corpus config not found: {default_seed}")
    
    # Ensure FTS5 is available
    if not check_fts5_available():
        logger.error("FTS5 is not available on this system")
        sys.exit(8)
    
    try:
        # Ensure corpus is fetched
        logger.info(f"Checking corpus cache for {root_ref}...")
        manifest = fetch_corpus(
            repo_url=repo_url,
            root_ref=root_ref,
            cache_dir=cache_dir,
            force_refresh=False,
        )
        logger.info(f"Using corpus at {manifest.local_path}")
        
        # Build full index
        logger.info(f"Building index for {root_ref}...")
        if seed_corpus:
            click.echo(f"[INFO] Indexing seed corpus only ({seed_corpus.name})")
        else:
            click.echo(f"[INFO] Indexing full corpus")
        
        result = build_full_index(
            manifest=manifest,
            output_dir=output_dir,
            window_lines=window_lines,
            overlap_lines=overlap_lines,
            seed_corpus_config=seed_corpus,
        )
        
        if result.get("status") != "success":
            error = result.get("error", "unknown")
            logger.error(f"Index build failed: {error}")
            if error == "fts5_unavailable":
                sys.exit(8)
            else:
                sys.exit(1)
        
        # Print results
        click.echo(f"\n[OK] Index created: {result['index_id']}")
        click.echo(f"  Corpus ID: {result['corpus_id']}")
        click.echo(f"  Root Ref: {root_ref}")
        click.echo(f"  Commit: {manifest.resolved_commit[:12]}")
        if seed_corpus:
            click.echo(f"  Scope: Seed corpus ({seed_corpus.name})")
        else:
            click.echo(f"  Scope: Full corpus")
        click.echo(f"  Chunks: {result['chunk_count']:,}")
        click.echo(f"  Files: {result['file_count']:,}")
        click.echo(f"  Retrieval Modes: {', '.join(result['retrieval_modes'])}")
        click.echo(f"  FTS DB: {result['fts_db_path']}")
        click.echo(f"  Manifest: {result['index_manifest_path']}")
        
        logger.info(
            f"Index ready: {result['index_id']}, "
            f"{result['chunk_count']} chunks, "
            f"{result['file_count']} files"
        )
        sys.exit(0)
    
    except InvalidRefError as e:
        logger.error(f"Invalid reference: {str(e)}")
        sys.exit(3)
    
    except Exception as e:
        logger.error(f"Index build failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("query", required=False)
@click.option(
    "--root-ref",
    required=False,
    help="Root reference for index resolution (overrides --index-id)",
)
@click.option(
    "--literal",
    "literal_query",
    required=False,
    help="Literal query text (supports flag-like literals such as -Y, --MuonBack, --MuDIS).",
)
@click.option(
    "--index-id",
    required=False,
    help="Explicit index ID to search",
)
@click.option(
    "--index-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory containing indexes (defaults to profile root).",
)
@click.option(
    "--profile",
    type=click.Choice(["root", "fairship", "project_docs"], case_sensitive=False),
    default="root",
    show_default=True,
    help="Named index profile.",
)
@click.option(
    "--top-k",
    type=int,
    default=10,
    help="Maximum number of results to return",
)
@click.option(
    "--retrieval-backend",
    type=click.Choice(["lexical", "semantic", "hybrid", "forest"], case_sensitive=False),
    default="lexical",
    help="Opt-in retrieval backend. Default is lexical. Forest is multi-chunk fusion.",
)
@click.option(
    "--baseline",
    is_flag=True,
    help="Force baseline lexical retrieval (shortcut for --retrieval-backend lexical).",
)
@click.option(
    "--retrieval-forest",
    help="Comma-separated profile names for retrieval forest. Default uses configs/retrieval_forest_profiles.json.",
)
@click.option(
    "--fusion",
    type=click.Choice(["rrf", "concat"], case_sensitive=False),
    default="rrf",
    help="Fusion method for retrieval forest.",
)
@click.option(
    "--dedup",
    type=click.Choice(["line_overlap", "none"], case_sensitive=False),
    default="line_overlap",
    help="Deduplication method for retrieval forest.",
)
@click.option(
    "--tie-breaker",
    type=click.Choice(["stable", "enhanced"], case_sensitive=False),
    default="stable",
    help="Tie-breaking logic for fused results.",
)
@click.option(
    "--semantic-manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional semantic manifest path for S1 semantic or hybrid search.",
)
@click.option(
    "--semantic-model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="Local embedding model name used for S1 semantic or hybrid search.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results as JSON",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def search(
    query: Optional[str],
    root_ref: str,
    literal_query: Optional[str],
    index_id: str,
    index_dir: Optional[Path],
    profile: str,
    top_k: int,
    retrieval_backend: str,
    baseline: bool,
    retrieval_forest: Optional[str],
    fusion: str,
    dedup: str,
    tie_breaker: str,
    semantic_manifest: Path,
    semantic_model: str,
    output_json: bool,
    verbose: bool,
):
    """Search indexed ROOT corpus for evidence.
    
    Performs lexical search by default, with opt-in S1 semantic or hybrid retrieval.
    Results are evidence candidates (file, line range, score) without answer generation.
    
    Examples:
        root-rag search "TTree::Draw" --root-ref v6-32-00
        root-rag search "RDataFrame" --index-id idx_abc123 --top-k 20
        root-rag search "Fill" --root-ref master --json
    
    Exit codes:
        0: Success with evidence found
        4: Index not found or not resolvable
        5: No evidence found for query
    """
    
    if verbose:
        logger.setLevel(logging.DEBUG)

    if query and literal_query:
        raise click.UsageError("Provide either QUERY or --literal, not both")
    effective_query = literal_query if literal_query is not None else query
    if not effective_query:
        raise click.UsageError("Missing query text. Pass QUERY or --literal <text>")

    try:
        indexes_root = _select_indexes_root(index_dir=index_dir, profile=profile.lower(), index_id=index_id)

        # Resolve index
        logger.debug(f"Resolving index: root_ref={root_ref}, index_id={index_id}")
        manifest = resolve_index(
            indexes_root=indexes_root,
            root_ref=root_ref,
            index_id=index_id,
        )
        logger.debug(f"Resolved to index {manifest.index_id}")
        
        # Load FTS5 database path
        db_path = manifest.fts_db_path
        if not db_path or not Path(db_path).exists():
            logger.error(f"FTS5 database not found: {db_path}")
            sys.exit(4)

        # Resolve backend and results
        results, actual_backend, fallback_reason = _resolve_backend_and_results(
            query=effective_query,
            manifest=manifest,
            top_k=top_k,
            retrieval_backend=retrieval_backend,
            retrieval_forest=retrieval_forest,
            fusion=fusion,
            dedup=dedup,
            tie_breaker=tie_breaker,
            baseline=baseline,
            profile=profile,
            semantic_manifest=semantic_manifest,
            semantic_model=semantic_model,
        )

        # Format and output results
        if output_json:
            output = {
                "results": [
                    {
                        "chunk_id": r.chunk_id,
                        "file_path": r.file_path,
                        "source_type": r.source_type,
                        "start_line": r.start_line,
                        "end_line": r.end_line,
                        "symbol_path": r.symbol_path,
                        "doc_origin": r.doc_origin,
                        "language": r.language,
                        "root_ref": r.root_ref,
                        "resolved_commit": r.resolved_commit,
                        "score": r.score,
                    }
                    for r in results
                ],
                "metadata": {
                    "actual_backend": actual_backend,
                    "fallback_reason": fallback_reason,
                    "fusion": fusion if actual_backend == "forest" else None,
                    "profile": profile,
                }
            }
            click.echo(json_module.dumps(output, indent=2))
        else:
            # Human-readable output
            click.echo(f"Actual backend: {actual_backend}")
            if fallback_reason:
                click.echo(f"Fallback reason: {fallback_reason}")
            click.echo("")
            
            for i, result in enumerate(results, 1):
                click.echo(
                    f"[{i}] [{result.source_type}] {result.file_path}:{result.start_line}-{result.end_line} "
                    f"score={result.score:.4f}"
                )
                if result.symbol_path:
                    click.echo(f"    Symbol: {result.symbol_path}")
                click.echo(f"    Doc: {result.doc_origin}")
                click.echo(f"    Commit: {result.resolved_commit[:12]}")
        
        logger.info(f"Found {len(results)} evidence candidates")
        sys.exit(0)
    
    except IndexNotFoundError as e:
        logger.error(f"Index not found: {str(e)}")
        sys.exit(4)
    
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command("build-semantic-index")
@click.option(
    "--root-ref",
    required=False,
    help="Root reference for index resolution (overrides --index-id)",
)
@click.option(
    "--index-id",
    required=False,
    help="Explicit index ID to augment with S1 semantic artifacts",
)
@click.option(
    "--index-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory containing indexes (defaults to profile root).",
)
@click.option(
    "--profile",
    type=click.Choice(["root", "fairship", "project_docs"], case_sensitive=False),
    default="root",
    show_default=True,
    help="Named index profile.",
)
@click.option(
    "--semantic-output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Optional explicit output directory for semantic artifacts",
)
@click.option(
    "--model-name",
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="Local sentence-transformers model name for S1 embeddings",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    help="Batch size for local embedding generation",
)
@click.option(
    "--device",
    default="cpu",
    help="Embedding inference device (default: cpu for reproducibility)",
)
def build_semantic_index(
    root_ref: str,
    index_id: str,
    index_dir: Optional[Path],
    profile: str,
    semantic_output_dir: Path,
    model_name: str,
    batch_size: int,
    device: str,
):
    """Build opt-in S1 semantic artifacts from an existing lexical index."""
    try:
        indexes_root = _select_indexes_root(index_dir=index_dir, profile=profile.lower(), index_id=index_id)
        manifest = resolve_index(
            indexes_root=indexes_root,
            root_ref=root_ref,
            index_id=index_id,
        )
        chunks_path = _resolve_chunks_path(manifest, indexes_root=indexes_root)
        if not chunks_path.exists():
            logger.error(f"Chunks file not found: {chunks_path}")
            sys.exit(4)

        output_dir = semantic_output_dir
        if output_dir is None:
            output_dir = Path(indexes_root) / manifest.index_id / "semantic" / _slugify_model_name(model_name)

        corpus_rows = load_corpus_rows(chunks_path)
        embedder = SentenceTransformerLocalEmbedder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
        semantic_manifest = build_semantic_index_artifacts(
            corpus_rows=corpus_rows,
            corpus_path=chunks_path,
            output_dir=output_dir,
            embedder=embedder,
            corpus_source_identifier=manifest.corpus_id,
        )

        manifest.semantic_manifest_path = str(output_dir / "semantic_manifest.json")
        retrieval_modes = list(manifest.retrieval_modes)
        for mode in ("semantic", "hybrid"):
            if mode not in retrieval_modes:
                retrieval_modes.append(mode)
        manifest.retrieval_modes = retrieval_modes
        manifest_path = Path(indexes_root) / manifest.index_id / "index_manifest.json"
        manifest.save(manifest_path)

        click.echo(f"[OK] S1 semantic artifacts created for {manifest.index_id}")
        click.echo(f"  Model: {semantic_manifest.model_name}")
        click.echo(f"  Dimension: {semantic_manifest.embedding_dimension}")
        click.echo(f"  Normalization: {semantic_manifest.normalization}")
        click.echo(f"  FAISS: {semantic_manifest.faiss_index_type}")
        click.echo(f"  Manifest: {output_dir / 'semantic_manifest.json'}")
        sys.exit(0)
    except IndexNotFoundError as e:
        logger.error(f"Index not found: {str(e)}")
        sys.exit(4)
    except Exception as e:
        logger.error(f"S1 semantic index build failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("query")
@click.option(
    "--root-ref",
    default="v6-36-08",
    help="Root reference for index resolution (default: v6-36-08 from FairShip)",
)
@click.option(
    "--index-id",
    required=False,
    help="Explicit index ID to search",
)
@click.option(
    "--index-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory containing indexes (defaults to profile root).",
)
@click.option(
    "--profile",
    type=click.Choice(["root", "fairship", "project_docs"], case_sensitive=False),
    default="root",
    show_default=True,
    help="Named index profile.",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Maximum number of results to return",
)
@click.option(
    "--retrieval-backend",
    type=click.Choice(["lexical", "forest"], case_sensitive=False),
    default="lexical",
    help="Opt-in retrieval backend. Default is lexical. Forest is multi-chunk fusion.",
)
@click.option(
    "--baseline",
    is_flag=True,
    help="Force baseline lexical retrieval (shortcut for --retrieval-backend lexical).",
)
@click.option(
    "--retrieval-forest",
    help="Comma-separated profile names for retrieval forest. Default uses configs/retrieval_forest_profiles.json.",
)
@click.option(
    "--fusion",
    type=click.Choice(["rrf", "concat"], case_sensitive=False),
    default="rrf",
    help="Fusion method for retrieval forest.",
)
@click.option(
    "--dedup",
    type=click.Choice(["line_overlap", "none"], case_sensitive=False),
    default="line_overlap",
    help="Deduplication method for retrieval forest.",
)
@click.option(
    "--tie-breaker",
    type=click.Choice(["stable", "enhanced"], case_sensitive=False),
    default="stable",
    help="Tie-breaking logic for fused results.",
)
def ask(query: str, root_ref: str, index_id: str, index_dir: Optional[Path], profile: str, top_k: int,
        retrieval_backend: str, baseline: bool, retrieval_forest: Optional[str], fusion: str, dedup: str, tie_breaker: str):
    """Ask a question and get evidence-based answers.
    
    Evidence-first retrieval: returns file paths and line ranges with ROOT version.
    Does not synthesize answers without evidence.
    
    Query syntax: Use keywords, not natural language.
    - Good: "TTree::Fill", "TGeoManager MakeBox", "TVector3 magnitude"
    - Avoid: "How do I fill a TTree?", "Where is the definition of?"
    - Stop words (filtered): where, what, how, is, are, the, definition, usage, etc.
    - Order doesn't matter: "TTree Fill" = "Fill TTree"
    
    For detailed query guide: See docs/QUERY_SYNTAX_GUIDE.md
    
    Examples:
        root-rag ask "TTree::Fill"
        root-rag ask "TH1F histogram"
        root-rag ask "TVector3 magnitude"
    
    Exit codes:
        0: Success with evidence found
        4: Index not found
        5: No evidence found
    """
    try:
        indexes_root = _select_indexes_root(index_dir=index_dir, profile=profile.lower(), index_id=index_id)
        # Resolve index
        manifest = resolve_index(
            indexes_root=indexes_root,
            root_ref=root_ref,
            index_id=index_id,
        )
        
        # Load FTS5 database
        db_path = manifest.fts_db_path
        if not db_path or not Path(db_path).exists():
            click.echo(f"Error: Index database not found for {root_ref}", err=True)
            sys.exit(4)
        
        # Resolve backend and results
        results, actual_backend, fallback_reason = _resolve_backend_and_results(
            query=query,
            manifest=manifest,
            top_k=top_k,
            retrieval_backend=retrieval_backend,
            retrieval_forest=retrieval_forest,
            fusion=fusion,
            dedup=dedup,
            tie_breaker=tie_breaker,
            baseline=baseline,
            profile=profile,
        )

        if not results:
            click.echo(f"Actual backend: {actual_backend}")
            if fallback_reason:
                click.echo(f"Fallback reason: {fallback_reason}")
            click.echo(f"No evidence found in ROOT {manifest.root_ref}")
            click.echo(f"Try broader search terms or check if the class is in the indexed corpus.")
            sys.exit(5)
        
        # Output evidence
        click.echo(f"Actual backend: {actual_backend}")
        if fallback_reason:
            click.echo(f"Fallback reason: {fallback_reason}")
        
        click.echo(f"Evidence (ROOT {manifest.root_ref}, commit {manifest.resolved_commit[:12]}):")
        # Logic to detect if forest was actually used
        if actual_backend == "forest":
             click.echo(f"Mode: Forest, Fusion: {fusion}, Dedup: {dedup}")
        elif actual_backend != "lexical":
             click.echo(f"Mode: {actual_backend}")
        click.echo("")
        
        for i, r in enumerate(results, 1):
            provenance = f" [{r.source_profile} rank {r.original_rank}]" if r.source_profile else ""
            click.echo(f"[{i}]{provenance} {r.file_path}:{r.start_line}-{r.end_line}")
            if r.symbol_path:
                click.echo(f"    Symbol: {r.symbol_path}")
        
        sys.exit(0)
        
    except IndexNotFoundError as e:
        click.echo(f"Error: No index found for {root_ref}", err=True)
        click.echo(f"Run: root-rag index --root-ref {root_ref}", err=True)
        sys.exit(4)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.argument("pattern")
@click.option(
    "--root-ref",
    default="v6-36-08",
    help="Root reference for index resolution (default: v6-36-08 from FairShip)",
)
@click.option(
    "--index-id",
    required=False,
    help="Explicit index ID to search",
)
@click.option(
    "--index-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory containing indexes (defaults to profile root).",
)
@click.option(
    "--profile",
    type=click.Choice(["root", "fairship", "project_docs"], case_sensitive=False),
    default="root",
    show_default=True,
    help="Named index profile.",
)
def grep(pattern: str, root_ref: str, index_id: str, index_dir: Optional[Path], profile: str):
    """Grep for symbol/pattern in indexed ROOT corpus.
    
    Fast exact-match search for symbols, methods, or code patterns.
    
    Examples:
        root-rag grep "TTree::Fill"
        root-rag grep "TGeoManager"
        root-rag grep "Draw"
    
    Exit codes:
        0: Success with matches found
        4: Index not found
        5: No matches found
    """
    try:
        indexes_root = _select_indexes_root(index_dir=index_dir, profile=profile.lower(), index_id=index_id)
        # Resolve index
        manifest = resolve_index(
            indexes_root=indexes_root,
            root_ref=root_ref,
            index_id=index_id,
        )
        
        # Load FTS5 database
        db_path = manifest.fts_db_path
        if not db_path or not Path(db_path).exists():
            click.echo(f"Error: Index database not found for {root_ref}", err=True)
            sys.exit(4)
        
        # Perform exact match search
        results = lexical_search(
            db_path=str(db_path),
            query=pattern,
            top_k=20,  # More results for grep
        )
        
        if not results:
            click.echo(f"No matches found for '{pattern}' in ROOT {manifest.root_ref}")
            sys.exit(5)
        
        # Output matches (grep-like format)
        click.echo(f"Matches in ROOT {manifest.root_ref} (commit {manifest.resolved_commit[:12]}):")
        click.echo("")
        
        for r in results:
            click.echo(f"{r.file_path}:{r.start_line}-{r.end_line}")
        
        click.echo(f"\n{len(results)} matches found")
        sys.exit(0)
        
    except IndexNotFoundError as e:
        click.echo(f"Error: No index found for {root_ref}", err=True)
        click.echo(f"Run: root-rag index --root-ref {root_ref}", err=True)
        sys.exit(4)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--index-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory containing indexes (defaults to profile root).",
)
@click.option(
    "--profile",
    type=click.Choice(["root", "fairship", "project_docs"], case_sensitive=False),
    default="root",
    show_default=True,
    help="Named index profile.",
)
def versions(index_dir: Optional[Path], profile: str):
    """List indexed ROOT versions.
    
    Shows all ROOT versions currently indexed and available for retrieval.
    
    Example:
        root-rag versions
    """
    index_dir = _select_indexes_root(index_dir=index_dir, profile=profile.lower(), index_id=None)
    
    if not index_dir.exists():
        click.echo("No indexes found. Run 'root-rag index' to create one.")
        sys.exit(0)
    
    # Find all index manifests
    manifests = []
    for manifest_file in index_dir.rglob("index_manifest.json"):
        try:
            manifest = IndexManifest.load(manifest_file)
            manifests.append(manifest)
        except Exception as e:
            logger.warning(f"Failed to load manifest {manifest_file}: {e}")
    
    if not manifests:
        click.echo("No indexes found. Run 'root-rag index --root-ref v6-36-08' to create one.")
        sys.exit(0)
    
    # Group by root_ref
    by_ref = {}
    for m in manifests:
        if m.root_ref not in by_ref:
            by_ref[m.root_ref] = []
        by_ref[m.root_ref].append(m)
    
    click.echo("Indexed ROOT versions:")
    click.echo("")
    
    for ref in sorted(by_ref.keys()):
        indexes = by_ref[ref]
        latest = max(indexes, key=lambda x: x.created_at)
        click.echo(f"  {ref}")
        click.echo(f"    Commit: {latest.resolved_commit[:12]}")
        click.echo(f"    Chunks: {latest.chunk_count:,}")
        click.echo(f"    Files: {latest.file_count:,}")
        click.echo(f"    Created: {latest.created_at}")
        click.echo(f"    Index ID: {latest.index_id}")
        click.echo("")
    
    sys.exit(0)


@main.command()
@click.argument("target")
@click.option(
    "--root-ref",
    required=False,
    help="Root reference for index resolution (overrides --index-id)",
)
@click.option(
    "--index-id",
    required=False,
    help="Explicit index ID to read from",
)
@click.option(
    "--index-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory containing indexes (defaults to profile root).",
)
@click.option(
    "--profile",
    type=click.Choice(["root", "fairship", "project_docs"], case_sensitive=False),
    default="root",
    show_default=True,
    help="Named index profile.",
)
@click.option(
    "--context",
    type=int,
    default=0,
    show_default=True,
    help="Extra context lines before/after requested range.",
)
def show(target: str, root_ref: str, index_id: str, index_dir: Optional[Path], profile: str, context: int):
    """Show indexed source lines for <file:start-end> from chunk text."""
    if context < 0:
        raise click.UsageError("--context must be >= 0")

    file_path, start_line, end_line = _parse_file_range(target)
    indexes_root = _select_indexes_root(index_dir=index_dir, profile=profile.lower(), index_id=index_id)

    try:
        manifest = resolve_index(
            indexes_root=indexes_root,
            root_ref=root_ref,
            index_id=index_id,
        )
    except IndexNotFoundError as e:
        logger.error(f"Index not found: {str(e)}")
        sys.exit(4)

    chunks_path = _resolve_chunks_path(manifest, indexes_root=indexes_root)
    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        sys.exit(4)

    found_file, line_map = _load_indexed_file_lines(chunks_path=chunks_path, file_path=file_path)
    if not found_file:
        click.echo(
            f"Error: file not present in indexed corpus/chunks: {file_path}",
            err=True,
        )
        sys.exit(5)

    from_line = max(1, start_line - context)
    to_line = end_line + context
    missing = 0
    width = len(str(to_line))

    click.echo(f"{file_path}:{start_line}-{end_line} (context={context})")
    for line_no in range(from_line, to_line + 1):
        text = line_map.get(line_no)
        if text is None:
            text = "[line not available in indexed chunks]"
            missing += 1
        click.echo(f"{line_no:>{width}} | {text}")

    if missing:
        click.echo("")
        click.echo(
            "[WARN] Indexed chunks do not cover all requested lines. "
            "Showing available chunk text only."
        )
    sys.exit(0)





if __name__ == '__main__':
    main()
