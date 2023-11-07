def regression_errors(y, yhat):
    preds = pd.DataFrame({
        'y': y,
        'yhat': yhat,
        'mean': y.mean()
    })
    
    squared_errors = (preds['y'] - preds['yhat']) ** 2
    # Sum of Squared Error, 
    sse = squared_errors.sum()
    mse = sse / len(preds)
    rmse = mse ** 0.5
    ess = ((preds['yhat'] - preds['mean']) **2).sum()
    tss = ess + sse
    return sse, ess, tss, mse, rmse

#------------------------------------------------------------
def baseline_mean_errors(y):
    preds = pd.DataFrame({
        'y': y,
        'mean': y.mean()
    })
    
    squared_errors_bl = (preds['y'] - preds['mean']) ** 2
    # Sum of Squared Error, 
    sse = squared_errors_bl.sum()
    mse = sse / len(preds)
    rmse = mse ** 0.5
    return sse, mse, rmse
#------------------------------------------------------------

def better_than_baseline(y, yhat):
    '''
    return a boolean for if the model beats the baseline prediction
    '''
    rmse_model = mean_squared_error(y, yhat, squared=False)
    sse, mse, rmse_baseline = baseline_mean_errors(y)
    return (rmse_model < rmse_baseline)