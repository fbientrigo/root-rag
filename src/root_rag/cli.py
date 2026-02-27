"""Root RAG CLI - Command line interface for root-rag."""
import logging
import sys
from pathlib import Path

import click

from root_rag.corpus import fetch_corpus, InvalidRefError

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


if __name__ == "__main__":
    main()
