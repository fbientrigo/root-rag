"""Core exception types for root-rag."""


class RootRagError(Exception):
    """Base exception for all root-rag errors."""


class InvalidRefError(RootRagError):
    """Raised when a git reference cannot be resolved."""


class GitOperationError(RootRagError):
    """Raised when a git operation fails."""


class CorpusError(RootRagError):
    """Raised when corpus operations fail."""


class ParserError(RootRagError):
    """Raised when file parsing fails."""


class ChunkingError(RootRagError):
    """Raised when chunking operation fails."""


class IndexBuildError(RootRagError):
    """Raised when index build fails."""


class IndexNotFoundError(RootRagError):
    """Raised when an index cannot be found or resolved."""


class RetrievalError(RootRagError):
    """Raised when retrieval operations fail."""
