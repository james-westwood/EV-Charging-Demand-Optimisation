"""1. Load features with load_features()                                                                                       
2. Define X and y — drop non-feature columns to get X, pull carbon_intensity as y
3. Call train_quantile_lgbm three times — P10, P50, P90                                                                     
4. Print a summary of the losses   """

from src.features.store import load_features
from src.models.forecasting.metrics import pinball_loss
from src.models.forecasting.trainer import train_quantile_lgbm


def train(feature_df):
    # Get the target column y
    y = feature_df["carbon_intensity"]

    # Drop the target column from X
    X = feature_df.drop("carbon_intensity", axis=1)
    
    # Also drop the datetime column because lgbm can't handle it
    X = X.drop("settlement_period", axis=1)

    alphas = [0.1, 0.5, 0.9]
    for alpha in alphas:
        model, oof = train_quantile_lgbm(X, y, alpha=alpha)
        loss = pinball_loss(alpha=alpha,
                            predictions=oof,
                            actuals=y)
        print(f"Alpha: {alpha}, Loss: {loss}")

if __name__ == "__main__":
    feature_df = load_features()
    train(feature_df)