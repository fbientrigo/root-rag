"""Smoke tests for root-rag CLI."""
import subprocess


def test_cli_help_returns_zero_exit_code():
    """Execute root-rag --help and verify it returns exit code 0."""
    result = subprocess.run(
        ["root-rag", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0