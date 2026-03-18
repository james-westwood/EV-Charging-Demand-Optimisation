import numpy as np

from src.models.forecasting.baselines import seasonal_naive_baseline, persistence_baseline
from src.models.forecasting.metrics import pinball_loss, calc_RMSE, calc_mae
from src.models.forecasting.trainer import train_quantile_lgbm


def evaluate_all_models(features_X,labels_y) -> dict:   
    
    X = features_X
    y = labels_y
    
    # For each of the alphas get a model trained 
    alphas = [0.1, 0.5, 0.9]
    metrics = {}
    
    for alpha in alphas:
        model, oof = train_quantile_lgbm(X, y, alpha=alpha)
        mask = ~np.isnan(oof)

        key = f"lgbm_p_{int(alpha * 100)}"
        metrics[key] = {} 
        
        metrics[key][f"pinball_{int(alpha * 100)}"] = pinball_loss(alpha=alpha,
                                                      predictions=oof[mask],
                                                      actuals=y.values[mask])
        if alpha == 0.5:
            metrics[key][f"mae_{int(alpha * 100)}"] = calc_mae(predictions=oof[mask],
                                                    actuals=y.values[mask])
            metrics[key][f"rmse_{int(alpha * 100)}"] = calc_RMSE(predictions=oof[mask],
                                                    actuals=y.values[mask])
    
    for name, (baseline, actuals) in (get_baseline_predictions(labels_y=labels_y).items()):
        metrics[name] = {} 
        metrics[name][f"pinball_{name}"] = pinball_loss(alpha=0.5,                                                          predictions=baseline,
                                                      actuals=actuals)
        metrics[name][f"mae_{name}"] = calc_mae(predictions=baseline,
                                                actuals=actuals)
        metrics[name][f"rmse_{name}"] = calc_RMSE(predictions=baseline,
                                                actuals=actuals)
      
    return metrics


def get_baseline_predictions(labels_y):
    
    # Get persistence baseline (-24hr) C intensity data for this period
    per_baseline = persistence_baseline(labels_y.values, h=48)
       
    # Get naive seasonal baseline (-336 settlement periods) for this period
    seas_baseline = seasonal_naive_baseline(labels_y.values, h=48, season=336)
    # return a labelled tuple
    return {"persistence": (per_baseline, labels_y.values[48:]),
            "seasonal_naive": (seas_baseline, labels_y.values[336:])}
 

if __name__ == "__main__":
    from src.features.store import load_features
    feature_df= load_features()
    features_X = feature_df.drop(["carbon_intensity", "settlement_period"], axis=1)  
    labels_y = feature_df["carbon_intensity"]
    metrics =evaluate_all_models(features_X,labels_y)
    print(metrics)