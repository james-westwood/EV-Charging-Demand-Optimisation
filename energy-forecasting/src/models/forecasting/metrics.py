import numpy as np


def pinball_loss(alpha, predictions, actuals) -> float:
    """The pinball loss function, also referred to as the quantile loss,
    is a metric used to assess the accuracy of a quantile forecast."""

    errors = actuals - predictions
    
    losses = np.where(errors >= 0, alpha * errors, (alpha - 1) * errors)
    
    return losses.mean() 
    
                                                                                                                                           
def calc_MAPE(predictions: np.ndarray, actuals: np.ndarray) -> float:                                                          
    """Mean Absolute Percentage Error (MAPE) used in 
    machine learning to evaluate the accuracy of 
    predictions made by models, particularly in 
    regression and forecasting tasks. """
    # Store APE values
    APE = []

    # Calculate APE for each record
    for i in range(len(actuals)):
        per_err = abs((actuals[i] - predictions[i]) / actuals[i])
        APE.append(per_err)
    
    # Calculate MAPE
    MAPE = sum(APE) / len(APE)
    
    return MAPE

def calc_RMSE(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE) used in machine learning to evaluate 
    the accuracy of predictions made by models, particularly in regression 
    and forecasting tasks. """
    return np.sqrt(np.mean((predictions - actuals) ** 2))