"""Root RAG CLI - Command line interface for root-rag."""
import logging
import sys
from pathlib import Path

import click

from root_rag.corpus import fetch_corpus, InvalidRefError
from root_rag.index import build_full_index, check_fts5_available
from root_rag.index.schemas import IndexManifest

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
    required=True,
    help="Branch, tag, or commit to index",
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
    "--config",
    type=click.Path(exists=False),
    default=None,
    help="Configuration file (reserved for future use)",
)
def index(
    root_ref: str,
    repo_url: str,
    cache_dir: Path,
    output_dir: Path,
    window_lines: int,
    overlap_lines: int,
    config: str,
):
    """Build FTS5 lexical search index for ROOT corpus.
    
    Creates SQLite FTS5 index for fast full-text search over ROOT code.
    Automatically fetches corpus if not cached.
    
    Examples:
        root-rag index --root-ref v6-32-00
        root-rag index --root-ref master --output-dir ~/.cache/root-rag/indexes
    
    Exit codes:
        0: Success
        1: Generic runtime failure
        2: Invalid CLI arguments
        3: Requested revision not found
        7: Configuration file error
        8: FTS5 not available on this system
    """
    
    # Validate configuration file if provided
    if config:
        config_path = Path(config)
        if not config_path.exists():
            logger.error(f"Config file not found: {config}")
            sys.exit(7)
        try:
            # Try to parse as JSON (basic validation)
            import json
            with open(config_path) as f:
                json.load(f)
        except Exception as e:
            logger.error(f"Invalid config file: {e}")
            sys.exit(7)
    
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
        result = build_full_index(
            manifest=manifest,
            output_dir=output_dir,
            window_lines=window_lines,
            overlap_lines=overlap_lines,
        )
        
        if result.get("status") != "success":
            error = result.get("error", "unknown")
            logger.error(f"Index build failed: {error}")
            if error == "fts5_unavailable":
                sys.exit(8)
            else:
                sys.exit(1)
        
        # Print results
        click.echo(f"[OK] Index created: {result['index_id']}")
        click.echo(f"  Corpus ID: {result['corpus_id']}")
        click.echo(f"  Root Ref: {root_ref}")
        click.echo(f"  Commit: {manifest.resolved_commit[:12]}")
        click.echo(f"  Schema Version: {result.get('created_at', 'N/A')}")
        click.echo(f"  Chunks: {result['chunk_count']}")
        click.echo(f"  Files: {result['file_count']}")
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


if __name__ == "__main__":
    main()
