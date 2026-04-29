

import sys

import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import pandas as pd  # import added here
import shap

from src.features.store import load_features


def get_latest_p50_run_id():
    """Retrieve the latest run_id for the P50 model from MLflow."""
    active_run = mlflow.active_run()
    if active_run:
        experiment_ids = [active_run.info.experiment_id]
    else:
        experiment_ids = ["0"]

    runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        filter_string="params.alpha = '0.5'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError("No MLflow runs found for P50 model (alpha=0.5)")
    return runs.iloc[0].run_id


def main():
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        run_id = get_latest_p50_run_id()
        print(f"Using latest run_id: {run_id}")

    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/lgbm_p50")

    features_df = load_features()

    features_df.drop("settlement_period", axis=1, inplace=True)
    features_df.drop("carbon_intensity", axis=1, inplace=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(features_df)

    # Beeswarm plot
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig("beeswarm.png", bbox_inches="tight")
    plt.clf()
    # Barplot
    shap.plots.bar(shap_values, show=False)
    plt.savefig("bar_plot.png")


if __name__ == "__main__":
    main()


def plot_waterfall(
    model,
    feature_row: dict,
    expected_value: float | None = None,
    save_path: str | None = None,
) -> None:
    """
    SHAP waterfall plot for a single prediction.

    Explains "Why did the model predict X at this time?"
    Walks through each feature's contribution.

    Args:
        model: trained LightGBM model
        feature_row: dict of feature_name -> value for one row
        expected_value: base value (E[f(x)]), auto-computed if None
        save_path: if given, save PNG there
    """
    explainer = shap.TreeExplainer(model)
    features_df = pd.DataFrame([feature_row])

    shap_values = explainer.shap_values(features_df)
    base_value = explainer.expected_value

    plt.figure()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=features_df.iloc[0],
        ),
        show=False,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    print(f"Base value (expected): {base_value}")
    print(f"Prediction: {base_value + shap_values[0].sum()}")
