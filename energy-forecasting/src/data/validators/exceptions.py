"""Custom exceptions for data validators."""


class ValidationError(Exception):
    """Raised when a DataFrame fails validation.

    Attributes:
        field: Name of the field that failed validation.
        message: Human-readable description of the failure.
    """

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(f"Validation failed for field '{field}': {message}")
