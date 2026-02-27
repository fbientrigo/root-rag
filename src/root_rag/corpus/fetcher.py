"""Corpus fetcher: resolve refs and manage corpus cache."""
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from root_rag.corpus.manifest import Manifest
from root_rag.core.errors import GitOperationError, InvalidRefError

logger = logging.getLogger(__name__)


def _get_repo_slug(repo_url: str) -> str:
    """Extract repository slug from URL.
    
    Examples:
        https://github.com/root-project/root.git -> root-project__root
        /local/path/to/repo -> repo
    """
    # Remove .git suffix if present
    clean_url = repo_url.rstrip("/")
    if clean_url.endswith(".git"):
        clean_url = clean_url[:-4]
    
    # Try to parse as URL
    parsed = urlparse(clean_url)
    if parsed.netloc:
        # URL like https://github.com/root-project/root
        path_parts = parsed.path.strip("/").split("/")
        return "__".join(path_parts[-2:]) if len(path_parts) >= 2 else path_parts[-1]
    else:
        # Local path
        return Path(clean_url).name


def resolve_git_ref(repo_url: str, root_ref: str) -> str:
    """Resolve a git reference to its commit SHA.
    
    Args:
        repo_url: Repository URL or local path
        root_ref: Reference to resolve (branch, tag, or commit)
    
    Returns:
        Resolved commit SHA (40-char hex string)
    
    Raises:
        InvalidRefError: If ref cannot be resolved
        GitOperationError: If git operation fails
    """
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--heads", "--tags", repo_url, root_ref],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            raise InvalidRefError(
                f"Cannot resolve ref '{root_ref}' in {repo_url}: {result.stderr.strip()}"
            )
        
        # Parse output: each line is "<sha>\t<ref>"
        lines = result.stdout.strip().split("\n")
        if not lines or not lines[0]:
            raise InvalidRefError(
                f"Reference '{root_ref}' not found in {repo_url}"
            )
        
        commit_sha = lines[0].split()[0]
        logger.info(f"Resolved {root_ref} → {commit_sha[:12]}")
        return commit_sha
    
    except subprocess.TimeoutExpired:
        raise GitOperationError(
            f"Git ls-remote timed out for {repo_url}"
        )
    except InvalidRefError:
        raise
    except Exception as e:
        raise GitOperationError(
            f"Failed to resolve ref {root_ref}: {str(e)}"
        )


def _clone_and_checkout(
    repo_url: str,
    root_ref: str,
    target_path: Path,
) -> str:
    """Clone repo to target_path and checkout root_ref.
    
    Returns:
        Resolved commit SHA
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clone
    logger.info(f"Cloning {repo_url} to {target_path}")
    result = subprocess.run(
        ["git", "clone", "--quiet", repo_url, str(target_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    
    if result.returncode != 0:
        raise GitOperationError(
            f"Failed to clone {repo_url}: {result.stderr.strip()}"
        )
    
    # Checkout ref
    logger.info(f"Checking out {root_ref}")
    result = subprocess.run(
        ["git", "-C", str(target_path), "checkout", "--quiet", root_ref],
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    if result.returncode != 0:
        raise InvalidRefError(
            f"Cannot checkout ref '{root_ref}': {result.stderr.strip()}"
        )
    
    # Get resolved commit SHA
    result = subprocess.run(
        ["git", "-C", str(target_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    if result.returncode != 0:
        raise GitOperationError(
            f"Failed to get commit SHA: {result.stderr.strip()}"
        )
    
    return result.stdout.strip()


def fetch_corpus(
    repo_url: str,
    root_ref: str,
    cache_dir: Path,
    tool_version: str = "0.0.1",
    force_refresh: bool = False,
) -> Manifest:
    """Fetch and cache a corpus, writing manifest.
    
    If corpus already exists in cache with same resolved_commit,
    returns existing manifest (idempotent). Otherwise, clones to tmpdir,
    checks out ref, and atomically moves to final cache location.
    
    Args:
        repo_url: Repository URL or local path
        root_ref: Reference to fetch (branch, tag, commit)
        cache_dir: Directory where corpora are cached
        tool_version: Version of root-rag creating this manifest
        force_refresh: If True, ignore cache and fetch fresh
    
    Returns:
        Manifest object with corpus metadata
    
    Raises:
        InvalidRefError: If ref cannot be resolved (exit code 3)
        GitOperationError: For other git failures
    """
    cache_dir = Path(cache_dir)
    
    # Resolve ref to commit SHA
    resolved_commit = resolve_git_ref(repo_url, root_ref)
    
    # Generate corpus_id: repo_slug__commit_short
    repo_slug = _get_repo_slug(repo_url)
    corpus_id = f"{repo_slug}__{resolved_commit[:12]}"
    corpus_path = cache_dir / corpus_id / "repo"
    manifest_path = cache_dir / corpus_id / "manifest.json"
    
    # Check if corpus already cached
    if corpus_path.exists() and manifest_path.exists() and not force_refresh:
        logger.info(f"Using cached corpus: {corpus_id}")
        return Manifest.load(manifest_path)
    
    # Clone to temporary directory
    with tempfile.TemporaryDirectory(prefix="root-rag-fetch-") as tmpdir:
        tmp_path = Path(tmpdir) / "repo"
        actual_commit = _clone_and_checkout(repo_url, root_ref, tmp_path)
        
        # Verify resolved_commit matches
        if actual_commit != resolved_commit:
            logger.warning(
                f"Resolved commit mismatch: {resolved_commit} vs {actual_commit}"
            )
        
        # Atomically move from tmpdir to final location
        logger.info(f"Installing corpus to {corpus_path}")
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove any partial/stale corpus
        if corpus_path.exists():
            shutil.rmtree(corpus_path)
        
        shutil.move(str(tmp_path), str(corpus_path))
    
    # Create and save manifest
    manifest = Manifest(
        repo_url=repo_url,
        root_ref=root_ref,
        resolved_commit=resolved_commit,
        local_path=str(corpus_path.absolute()),
        fetched_at=datetime.now(timezone.utc).isoformat(),
        dirty=False,
        tool_version=tool_version,
    )
    
    manifest.save(manifest_path)
    logger.info(f"Manifest saved to {manifest_path}")
    
    return manifest
