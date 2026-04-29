import numpy as np


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


def plot_calibration(
    calibration_results: dict[str, float],
    save_path: str | None = None,
) -> None:
    """
    Create a calibration plot: claimed quantile vs observed frequency.

    Args:
        calibration_results: output of calculate_calibration
        save_path: if given, save PNG there; otherwise renders interactively
    """
    import matplotlib.pyplot as plt

    claimed = [
        calibration_results["p10_claimed"],
        calibration_results["p50_claimed"],
        calibration_results["p90_claimed"],
    ]
    observed = [
        calibration_results["p10_observed"],
        calibration_results["p50_observed"],
        calibration_results["p90_observed"],
    ]

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "r--", alpha=0.5, label="perfect")
    plt.scatter(claimed, observed, s=100, alpha=0.7)
    plt.xlabel("Claimed quantile")
    plt.ylabel("Observed frequency")
    plt.title("Quantile calibration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()