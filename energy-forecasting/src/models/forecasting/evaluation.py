from src.models.forecasting.baselines import seasonal_naive_baseline, persistence_baseline
from src.models.forecasting.metrics import pinball_loss, calc_RMSE, calc_MAPE
from src.models.forecasting.trainer import train_quantile_lgbm


def evaluate_all_models(features_X,labels_y) -> dict:   
    
    X = features_X
    y = labels_y
    
    # For each of the alphas get a model trained 
    alphas = [0.1, 0.5, 0.9]
    metrics = {}
    for alpha in alphas:
        model, oof = train_quantile_lgbm(X, y, alpha=alpha)
    
    
        metrics[f"pinball_{alpha*10}"] = pinball_loss(alpha=alpha,
                                                      predictions=oof,
                                                      actuals=y)
        metrics[f"mape_{alpha*10}"] = calc_MAPE(predictions=oof,
                                                actuals=y)
        metrics[f"rmse_{alpha*10}"] = calc_RMSE(predictions=oof,
                                                actuals=y)
        
    return {"quantile_lgbm": metrics**}


def get_baseline_predictions():
    
    predictions = {}
    
    # Get persistence baseline (-24hr) C intensity data for this period
    predictions["per_baseline"] = ....
       
    # Get naive seasonal baseline (-336 settlement periods) for this period
    predictions["seas_baseline"] = ...
    
    