import numpy as np


def check_quantile_monotonicity(                                                                                                                
      p10: np.ndarray,
      p50: np.ndarray,                                                                                                                            
      p90: np.ndarray,
  ) -> dict[str, float]:
    """Checks monotonicity of predictions. A function is
    monotonic if, as the input values increase, the output
    values either always increase or always decrease"""
                                                                                                                                       
    if not len(p10) == len(p50) == len(p90):
        raise ValueError("p10, p50, and p90 must have the same length")
    
    violation_count = np.where(
        (p10 > p50) | (p50 > p90),
        1,
        0,
    ).sum()
      
    violation_pct = violation_count/len(p10)
    
    return {"violation_count": violation_count, "violation_pct": violation_pct} 