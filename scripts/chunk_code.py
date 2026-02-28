#!/usr/bin/env python
"""CLI script to generate chunks.jsonl from a corpus manifest."""
import logging
import sys
from pathlib import Path

import click

from root_rag.corpus import Manifest
from root_rag.index.builder import build_index

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
)
logger = logging.getLogger("chunk_code")


@click.command()
@click.option(
    "--manifest",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to manifest.json file",
)
@click.option(
    "--out",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("data/processed/chunks"),
    help="Output directory for chunks.jsonl (default: data/processed/chunks)",
)
@click.option(
    "--window-lines",
    type=int,
    default=80,
    help="Lines per window/chunk (default: 80)",
)
@click.option(
    "--overlap-lines",
    type=int,
    default=10,
    help="Lines of overlap between windows (default: 10)",
)
def main(manifest: Path, out: Path, window_lines: int, overlap_lines: int):
    """Generate chunks.jsonl from a corpus manifest.
    
    Example:
        python scripts/chunk_code.py \\
            --manifest data/raw/corpora/root__abc123de/manifest.json \\
            --out data/processed/chunks
    
    Output:
        Writes chunks to: data/processed/chunks/<corpus_id>/chunks.jsonl
    """
    try:
        # Load manifest
        logger.info(f"Loading manifest from {manifest}")
        manifest_obj = Manifest.load(manifest)
        
        logger.info(f"Corpus: {manifest_obj.root_ref}")
        logger.info(f"Commit: {manifest_obj.resolved_commit}")
        logger.info(f"Path: {manifest_obj.local_path}")
        
        # Build index
        logger.info(f"Building index with window={window_lines}, overlap={overlap_lines}")
        result = build_index(
            manifest=manifest_obj,
            output_dir=out,
            window_lines=window_lines,
            overlap_lines=overlap_lines,
        )
        
        if result.get("status") != "success":
            logger.error(f"Build failed: {result.get('error', 'unknown error')}")
            sys.exit(1)
        
        # Print summary
        logger.info(f"Chunks: {result['chunk_count']}")
        logger.info(f"Files: {result['file_count']}")
        logger.info(f"Output: {result['chunks_path']}")
        
        click.echo(
            f"\n[OK] Generated {result['chunk_count']} chunks from "
            f"{result['file_count']} files"
        )
        click.echo(f"Chunks written to: {result['chunks_path']}")
        
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        click.echo(f"[ERROR] {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
