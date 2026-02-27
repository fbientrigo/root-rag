"""Core exception types for root-rag."""


class RootRagError(Exception):
    """Base exception for all root-rag errors."""
    pass


class InvalidRefError(RootRagError):
    """Raised when a git reference cannot be resolved."""
    pass


class GitOperationError(RootRagError):
    """Raised when a git operation fails."""
    pass


class CorpusError(RootRagError):
    """Raised when corpus operations fail."""
    pass
