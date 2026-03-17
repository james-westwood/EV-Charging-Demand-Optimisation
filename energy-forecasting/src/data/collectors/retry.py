import functools
import logging
import time
from typing import Any, Callable, Protocol, Tuple, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CallableWithName(Protocol[T]):
    """Protocol for callable objects that have a __name__ attribute."""

    __name__: str

    def __call__(self, *args: Any, **kwargs: Any) -> T: ...


def retry(
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> Callable[[CallableWithName[T]], Callable[..., T]]:
    """Decorator for retrying a function with exponential backoff.

    Args:
        exceptions: The exception(s) to catch and retry on.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay for exponential backoff (2^n * base_delay).
    """

    def decorator(func: CallableWithName[T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) reached for {func.__name__}. "
                            f"Last error: {str(e)}"
                        )
                        raise last_exception

                    delay = (2**attempt) * base_delay
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                        f"{func.__name__}: {str(e)}. Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)

            # This line is unreachable because the loop always either returns or raises
            raise RuntimeError("Unexpected end of retry loop")

        return wrapper

    return decorator
