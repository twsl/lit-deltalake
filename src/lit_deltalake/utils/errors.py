class DeltaDatasetError(Exception):
    """Base exception for Delta dataset failures."""


class BackendUnavailableError(DeltaDatasetError, ImportError):
    """Raised when an optional reader dependency is not installed."""


class InvalidDatasetConfigurationError(DeltaDatasetError, ValueError):
    """Raised when dataset configuration is internally inconsistent."""


class UnsupportedReaderOperationError(DeltaDatasetError, NotImplementedError):
    """Raised when a reader cannot perform a requested operation."""
