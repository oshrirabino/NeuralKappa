"""Custom exceptions for NeuralKappa domain validation."""


class NeuralKappaError(ValueError):
    """Base class for domain errors raised by NeuralKappa."""


class InvalidISIError(NeuralKappaError):
    """Raised when ISI inputs are invalid (non-finite/non-positive/wrong shape)."""


class InsufficientDataError(NeuralKappaError):
    """Raised when there are not enough samples to compute a metric."""


class DomainError(NeuralKappaError):
    """Raised when a metric denominator or domain is undefined."""
