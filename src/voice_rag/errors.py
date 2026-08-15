class RAGError(Exception):
    """Expected, user-safe pipeline error."""


class ProviderError(RAGError):
    """External provider failed after retries."""


class IndexNotLoadedError(RAGError):
    """The API started without a built index."""
