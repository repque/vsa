"""Custom exceptions for VSA application."""


class VSAException(Exception):
    """Base exception for VSA application."""
    pass


class DataFetchError(VSAException):
    """Raised when data fetching fails."""
    pass


class ModelError(VSAException):
    """Raised when model operations fail."""
    pass


class ConfigurationError(VSAException):
    """Raised when configuration is invalid or missing."""
    pass


class PositionError(VSAException):
    """Raised when position operations fail."""
    pass


class EmailError(VSAException):
    """Raised when email operations fail."""
    pass
