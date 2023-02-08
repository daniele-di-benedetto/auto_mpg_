import numpy
import pandas
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score


def VIF(X: numpy.ndarray, columns: list) -> None:
    """Compute the predictor's variance inflation factor.
    
    Parameters
    ----------
    X : np.array
        matrix with predictors
    columns : list
        list of predictors
        
    Returns
    -------
        VIF for each predictor
    
    """
    for index in range(len(columns)):
        vif = variance_inflation_factor(numpy.matrix(X), index)
        print(f"Variance Inflation Factor for {columns[index]}: {round(number =vif, ndigits=2)}")


def regression_metrics(y_true, y_pred) -> None:
    """
    Get performance metrics of fitted regression model
    
    Parameters
    ----------
    y_true : np.array
        true labels
    y_pred : np.array
        predicted values by the model
        
    Returns
    -------
        regression metrics
    """
    mean_ae = mean_absolute_error(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = numpy.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean absolute error: {round(number=mean_ae, ndigits=2)}")
    print(f"Median absolute error: {round(number=median_ae, ndigits=2)}")
    print(f"Mean squared error: {round(number=mse, ndigits=2)}")
    print(f"Root mean squared error: {round(number=rmse, ndigits=2)}")
    print(f"R2 score: {round(number=r2, ndigits=2)}")


def compute_F_statistic(X, y, model):
    """
    Parameters
    ----------
    X : np.array
        predictors
    y : np.array
        response variable
    model : sklearn model
        regression model
    
    Returns
    -------
        F-statistic and p-value
    """
    RSS = numpy.sum((y - model.predict(X))**2)
    TSS = numpy.sum((numpy.mean(y) - y)**2)
    F_statistic = ((TSS - RSS) / X.shape[1]) / (RSS / (X.shape[0] - X.shape[1] - 1))
    F_pvalue = stats.f.sf(F_statistic, X.shape[1] - 1, X.shape[0] - X.shape[1])
    
    print(f"F-statistic: {F_statistic:.2f}")
    print(f"p-value: {F_pvalue: .2f}")