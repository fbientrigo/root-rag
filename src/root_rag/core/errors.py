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


class ParserError(RootRagError):
    """Raised when file parsing fails."""
    pass


class ChunkingError(RootRagError):
    """Raised when chunking operation fails."""
    pass


class IndexBuildError(RootRagError):
    """Raised when index build fails."""
    pass
