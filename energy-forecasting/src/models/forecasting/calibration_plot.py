import numpy as np
import plotly.graph_objects as go


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
) -> go.Figure:
    """
    Create a calibration plot: claimed quantile vs observed frequency.

    Args:
        calibration_results: output of calculate_calibration
        save_path: if given, save HTML there; otherwise returns figure

    Returns:
        Plotly figure
    """
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

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(dash="dash", color="red"),
        name="perfect",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=claimed,
        y=observed,
        mode="markers",
        marker=dict(size=14),
        text=[f"P{int(c*100)}" for c in claimed],
        hovertemplate="Claimed: %{x}<br>Observed: %{y}<extra>%{text}</extra>",
    ))

    fig.update_layout(
        xaxis_title="Claimed quantile",
        yaxis_title="Observed frequency",
        title="Quantile calibration",
        template="plotly_white",
        width=500,
        height=500,
    )

    if save_path:
        fig.write_html(save_path)
    return fig


def plot_uncertainty_bands(
    settlement_periods: np.ndarray,
    actuals: np.ndarray,
    p10_preds: np.ndarray,
    p50_preds: np.ndarray,
    p90_preds: np.ndarray,
    n_points: int = 500,
    save_path: str | None = None,
) -> go.Figure:
    """
    Plot forecast uncertainty bands with actuals overlaid.

    Shows P10-P90 shaded region, P50 line, and actuals as points.
    Visual complement to numerical calibration metrics.

    Args:
        settlement_periods: datetime array for x-axis
        actuals: actual values
        p10_preds: P10 predictions (lower bound)
        p50_preds: P50 predictions (median)
        p90_preds: P90 predictions (upper bound)
        n_points: how many points to plot (for readability)
        save_path: if given, save HTML there

    Returns:
        Plotly figure
    """
    step = max(1, len(actuals) // n_points)
    idx = np.arange(0, len(actuals), step)

    x = settlement_periods[idx]
    y_p10 = p10_preds[idx]
    y_p90 = p90_preds[idx]
    y_p50 = p50_preds[idx]
    y_actual = actuals[idx]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([y_p90, y_p10[::-1]]),
        fill="toself",
        fillcolor="rgba(100, 149, 237, 0.3)",
        line=dict(color="rgba(100, 149, 237, 0.3)"),
        name="P10-P90 band",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=y_p50,
        mode="lines",
        line=dict(color="blue", width=1),
        name="P50",
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=y_actual,
        mode="markers",
        marker=dict(size=4, color="red", opacity=0.6),
        name="Actual",
    ))

    fig.update_layout(
        xaxis_title="Settlement period",
        yaxis_title="Carbon intensity (gCO2/kWh)",
        title="Forecast uncertainty bands",
        template="plotly_white",
        width=900,
        height=400,
        hovermode="x unified",
    )

    if save_path:
        fig.write_html(save_path)
    return fig


def plot_pinball_comparison(
    pinball_losses: dict[str, dict[str, float]],
    save_path: str | None = None,
) -> go.Figure:
    """
    Grouped bar chart: pinball loss by quantile for different models.

    Shows LightGBM P10/P50/P90 vs persistence vs seasonal naive.
    Makes the "model beats baselines" story visual.

    Args:
        pinball_losses: {
              "lgbm": {"p10": 10.5, "p50": 8.2, "p90": 12.1},
              "persistence": {"p10": 15.3, "p50": 14.8, "p90": 18.2},
              "seasonal_naive": {"p10": 12.1, "p50": 11.5, "p90": 14.0},
            }
        save_path: if given, save HTML there

    Returns:
        Plotly figure
    """
    models = list(pinball_losses.keys())
    quantiles = ["p10", "p50", "p90"]

    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, model in enumerate(models):
        values = [pinball_losses[model][q] for q in quantiles]
        fig.add_trace(go.Bar(
            name=model,
            x=quantiles,
            y=values,
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        xaxis_title="Quantile",
        yaxis_title="Pinball loss",
        title="Pinball loss by quantile",
        template="plotly_white",
        barmode="group",
        width=700,
        height=450,
    )

    if save_path:
        fig.write_html(save_path)
    return fig