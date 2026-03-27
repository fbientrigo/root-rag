"""Pytest fixtures for root-rag tests."""
import subprocess
from pathlib import Path
from typing import Dict

import pytest


@pytest.fixture
def git_repo_fixture(tmp_path: Path) -> Dict[str, any]:
    """Create a minimal git repository with tag and branch.
    
    Returns dict with:
        - path: Path to repo
        - tag_sha: SHA of v0.1 tag
        - dev_sha: SHA of dev branch
        - main_sha: SHA of main branch
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    
    # Initialize repo
    subprocess.run(
        ["git", "init"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    
    # Configure git user
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    
    # Create initial file and commit
    (repo_path / "A.h").write_text("#include <iostream>\n")
    subprocess.run(
        ["git", "add", "A.h"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    
    # Get main branch SHA
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    main_sha = result.stdout.strip()
    
    # Create tag v0.1
    subprocess.run(
        ["git", "tag", "v0.1"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    tag_sha = main_sha  # Same as main
    
    # Create dev branch
    subprocess.run(
        ["git", "checkout", "-b", "dev"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    
    # Add another file on dev
    (repo_path / "B.h").write_text("#include <vector>\n")
    subprocess.run(
        ["git", "add", "B.h"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Add B.h on dev"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    
    # Get dev branch SHA
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    dev_sha = result.stdout.strip()
    
    return {
        "path": repo_path,
        "tag_sha": tag_sha,
        "dev_sha": dev_sha,
        "main_sha": main_sha,
    }


@pytest.fixture
def cpp_repo_fixture(tmp_path: Path) -> Dict[str, any]:
    """Create a git repository with C++ header and implementation files.
    
    Returns dict with:
        - path: Path to repo
        - resolved_commit: SHA of HEAD commit
        - root_ref: "main" (branch name)
        - files: {"tree.h": path, "tree.cxx": path}
    """
    repo_path = tmp_path / "cpp_repo"
    repo_path.mkdir()
    
    # Initialize repo
    subprocess.run(
        ["git", "init"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    
    # Configure git user
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    
    # Create directory structure
    inc_dir = repo_path / "tree" / "inc"
    inc_dir.mkdir(parents=True)
    src_dir = repo_path / "tree" / "src"
    src_dir.mkdir(parents=True)
    
    # Create header file with Doxygen comment
    header_content = '''/**
 * @brief TTree class for ROOT
 * @file TTree.h
 * @author ROOT Team
 */

#ifndef ROOT_TTREE_H
#define ROOT_TTREE_H

class TTree {
public:
    /** \\brief Draw histogram or expression
     * \\param expression Expression to draw
     * \\param selection Selection criteria
     * \\return Number of entries drawn
     */
    virtual Long64_t Draw(const char* expression, const char* selection = "");
    
    virtual ~TTree();
};

#endif
'''
    tree_h = inc_dir / "TTree.h"
    tree_h.write_text(header_content)
    
    # Create implementation file
    impl_content = '''#include "TTree.h"

Long64_t TTree::Draw(const char* expression, const char* selection) {
    // Implementation of Draw
    // This would do actual drawing logic
    return 0;
}

TTree::~TTree() {
    // Cleanup
}
'''
    tree_cxx = src_dir / "TTree.cxx"
    tree_cxx.write_text(impl_content)
    
    # Commit files
    subprocess.run(
        ["git", "add", "tree"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit with TTree"],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )
    
    # Get commit SHA
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    resolved_commit = result.stdout.strip()
    
    return {
        "path": repo_path,
        "resolved_commit": resolved_commit,
        "root_ref": "main",
        "files": {
            "tree.h": tree_h,
            "tree.cxx": tree_cxx,
        },
    }
