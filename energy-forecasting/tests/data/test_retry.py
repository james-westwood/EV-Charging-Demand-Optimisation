import pytest
import time
from unittest.mock import MagicMock, patch
from src.data.collectors.retry import retry

def test_retry_success_after_two_failures():
    """
    Test that the retry decorator calls the function 3 times total
    when it fails twice then succeeds.
    """
    mock_func = MagicMock()
    # Fails twice (Exception), then succeeds (return 'Success')
    mock_func.side_effect = [Exception("Failure 1"), Exception("Failure 2"), "Success"]
    
    @retry(max_retries=3, base_delay=1)
    def decorated_func():
        return mock_func()
    
    with patch("time.sleep") as mock_sleep:
        result = decorated_func()
        
    assert result == "Success"
    assert mock_func.call_count == 3
    # Check if sleep was called twice with 2^0 * 1 and 2^1 * 1
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)

def test_retry_failure_after_exhaustion():
    """
    Test that the retry decorator raises the original exception
    after failing max_retries + 1 times.
    """
    mock_func = MagicMock()
    mock_func.side_effect = Exception("Final Failure")
    
    @retry(max_retries=3, base_delay=1)
    def decorated_func():
        return mock_func()
    
    with patch("time.sleep") as mock_sleep:
        with pytest.raises(Exception, match="Final Failure"):
            decorated_func()
            
    # Function is called once initially + 3 retries = 4 calls total
    assert mock_func.call_count == 4
    # Sleep is called 3 times (after failure 1, 2, and 3)
    assert mock_sleep.call_count == 3
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)
    mock_sleep.assert_any_call(4.0)

def test_retry_specific_exception():
    """
    Test that only specified exceptions trigger a retry.
    """
    class CustomError(Exception):
        pass
        
    mock_func = MagicMock()
    mock_func.side_effect = ValueError("Wrong error")
    
    @retry(exceptions=(CustomError,), max_retries=3, base_delay=1)
    def decorated_func():
        return mock_func()
        
    with patch("time.sleep") as mock_sleep:
        with pytest.raises(ValueError, match="Wrong error"):
            decorated_func()
            
    # ValueError is not in (CustomError,), so it should fail immediately
    assert mock_func.call_count == 1
    assert mock_sleep.call_count == 0
