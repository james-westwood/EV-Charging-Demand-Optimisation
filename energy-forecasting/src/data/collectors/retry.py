import functools
import time
import logging
from typing import Callable, Any, Type, Tuple, Union

logger = logging.getLogger(__name__)

def retry(
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    max_retries: int = 3,
    base_delay: float = 2.0
) -> Callable:
    """
    Decorator for retrying a function with exponential backoff.
    
    Args:
        exceptions: The exception(s) to catch and retry on.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay for exponential backoff (2^n * base_delay).
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
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
                    
                    delay = (2 ** attempt) * base_delay
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
            
            # This part should ideally not be reached because of the raise in the loop
            if last_exception:
                raise last_exception
        return wrapper
    return decorator
