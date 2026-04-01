"""Root RAG CLI - Command line interface for root-rag."""
import json as json_module
import logging
import sys
from pathlib import Path

import click

from root_rag.corpus import fetch_corpus, InvalidRefError
from root_rag.core.errors import IndexNotFoundError
from root_rag.index import build_full_index, check_fts5_available
from root_rag.index.locator import resolve_index
from root_rag.index.schemas import IndexManifest
from root_rag.retrieval import lexical_search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
)
logger = logging.getLogger("root_rag")


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
@click.argument("query")
@click.option(
    "--root-ref",
    required=False,
    help="Root reference for index resolution (overrides --index-id)",
)
@click.option(
    "--index-id",
    required=False,
    help="Explicit index ID to search",
)
@click.option(
    "--index-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/indexes"),
    help="Directory containing indexes",
)
@click.option(
    "--top-k",
    type=int,
    default=10,
    help="Maximum number of results to return",
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
    query: str,
    root_ref: str,
    index_id: str,
    index_dir: Path,
    top_k: int,
    output_json: bool,
    verbose: bool,
):
    """Search indexed ROOT corpus for evidence.
    
    Performs lexical full-text search over an indexed ROOT corpus revision.
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
    
    try:
        # Resolve index
        logger.debug(f"Resolving index: root_ref={root_ref}, index_id={index_id}")
        manifest = resolve_index(
            indexes_root=index_dir,
            root_ref=root_ref,
            index_id=index_id,
        )
        logger.debug(f"Resolved to index {manifest.index_id}")
        
        # Load FTS5 database path
        db_path = manifest.fts_db_path
        if not db_path or not Path(db_path).exists():
            logger.error(f"FTS5 database not found: {db_path}")
            sys.exit(4)
        
        logger.debug(f"Searching in {db_path}")
        
        # Perform lexical search
        results = lexical_search(
            db_path=str(db_path),
            query=query,
            top_k=top_k,
        )
        
        # Handle no results
        if not results:
            logger.error(f"No evidence found for query: {query}")
            sys.exit(5)
        
        # Format and output results
        if output_json:
            output = [
                {
                    "chunk_id": r.chunk_id,
                    "file_path": r.file_path,
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
            ]
            click.echo(json_module.dumps(output, indent=2))
        else:
            # Human-readable output
            for i, result in enumerate(results, 1):
                click.echo(
                    f"[{i}] {result.file_path}:{result.start_line}-{result.end_line} "
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
    default=Path("data/indexes"),
    help="Directory containing indexes",
)
@click.option(
    "--top-k",
    type=int,
    default=5,
    help="Maximum number of results to return",
)
def ask(query: str, root_ref: str, index_id: str, index_dir: Path, top_k: int):
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
        # Resolve index
        manifest = resolve_index(
            indexes_root=index_dir,
            root_ref=root_ref,
            index_id=index_id,
        )
        
        # Load FTS5 database
        db_path = manifest.fts_db_path
        if not db_path or not Path(db_path).exists():
            click.echo(f"Error: Index database not found for {root_ref}", err=True)
            sys.exit(4)
        
        # Perform search
        results = lexical_search(
            db_path=str(db_path),
            query=query,
            top_k=top_k,
        )
        
        if not results:
            click.echo(f"No evidence found in ROOT {manifest.root_ref}")
            click.echo(f"Try broader search terms or check if the class is in the indexed corpus.")
            sys.exit(5)
        
        # Output evidence
        click.echo(f"Evidence (ROOT {manifest.root_ref}, commit {manifest.resolved_commit[:12]}):")
        click.echo("")
        
        for i, r in enumerate(results, 1):
            click.echo(f"[{i}] {r.file_path}:{r.start_line}-{r.end_line}")
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
    default=Path("data/indexes"),
    help="Directory containing indexes",
)
def grep(pattern: str, root_ref: str, index_id: str, index_dir: Path):
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
        # Resolve index
        manifest = resolve_index(
            indexes_root=index_dir,
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
    default=Path("data/indexes"),
    help="Directory containing indexes",
)
def versions(index_dir: Path):
    """List indexed ROOT versions.
    
    Shows all ROOT versions currently indexed and available for retrieval.
    
    Example:
        root-rag versions
    """
    index_dir = Path(index_dir)
    
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





if __name__ == '__main__':
    main()
