


def calculate_calibration(
    actuals: np.ndarray,
    p10_preds: np.ndarray,
    p50_preds: np.ndarray,
    p90_preds: np.ndarray,
) -> dict[str, float]:
    """
    Returns a dict with keys like:
    - 'p10_claimed': 0.1
    - 'p10_observed': 0.12  (or whatever you calculate)
    - 'p90_claimed': 0.9
    - 'p90_observed': 0.87  (or whatever you calculate)
    """
    
    # p10 means 10% of actuals should be below the P10 predictions
    p10_observed = (np.sum(actuals < p10_preds)) / len(actuals)
    
    # p50 means 50% of actuals should be below the P50 predictions
    p50_observed = (np.sum(actuals < p50_preds)) / len(actuals)
    
    # p90 means 90% of actuals should be below the P90 predictions
    p90_observed = (np.sum(actuals < p90_preds)) / len(actuals)

    p10_claimed = 0.1
    p50_claimed = 0.5
    p90_claimed = 0.9

    return {
            "p10_claimed": p10_claimed,  # This is just the input value (0.1 for P10)
            "p10_observed": p10_observed,  # Replace with your calculation
            "p50_claimed": p50_claimed,
            "p50_observed": p50_observed,  # Replace with your calculation
            "p90_claimed": p90_claimed,
            "p90_observed": p90_observed,  # Replace with your calculation
        }
# Questions to guide your implementation:
# 1. How do you calculate what % of actuals fell below P10 predictions? (Hint: compare two arrays, count Trues, divide by length)
# 2. Why do we need the mask = ~np.isnan(oof_preds) pattern from trainer.py:63? What happens to early rows?
# 3. What does "observed frequency" mean when P10 claimed 0.1 and you calculate 0.15? Is the model overconfident or underconfident?