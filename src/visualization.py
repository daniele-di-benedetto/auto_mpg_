import pandas
import numpy
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
import random

def get_random_color(palette='hls', n=20):
    '''
    Get a random color from a given palette.
    
    Parameters
    ----------
    palette : str, default : 'hls'
        Palette code. For more palette check the seaborn documentation.
    
    n : int, default : 10
        Number of colors in the palette.
        
    References
    ----------
    https://seaborn.pydata.org/tutorial/color_palettes.html#general-principles-for-using-color-in-plots
    '''
    return sns.color_palette("hls", n)[random.randint(0, n - 1)]


def regression_diagnostic_plot(y_true: numpy.array, y_pred: numpy.array) -> None:
    """Create diagnostic plots for regression models.
    
    Parameters
    ----------
    y_true : numpy.array
        target values
    y_pred : numpy.array
        predicted values
    
    Returns
    -------
    None
        residuals plot
    
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    residuals = y_true - y_pred
    xmin, xmax = y_pred.min(), y_pred.max()
    fig = plt.figure(figsize=(15, 5))
    ax = fig.subplots(nrows=1, ncols=2)
    # Residuals plot
    ax[0].set(title="Residuals vs. fitted plot", xlabel="Fitted values", ylabel="Residuals")
    ax[0].hlines(y=0, xmin=xmin, xmax=xmax, colors="red", linestyles="--", linewidth=2)
    sns.scatterplot(x=y_pred, y=residuals, ax=ax[0])
    ax[0].grid(True)
    # Q-Q plot 
    ax[1].set_title("Q-Q plot of residuals")
    qqplot(data=residuals, line="45", fit="True", markersize=5, ax=ax[1])
    ax[1].grid(True)
    plt.tight_layout()
    plt.show()


def plot_dataframe_corr(data : pandas.DataFrame) -> None:
    """Create correlation plot between numerical variables in a given dataframe.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame


    Returns
    -------
    None
        correlation plot
    
    """
    data_corr = data.corr(numeric_only=True)
    trimask = numpy.triu(data_corr)
    plt.figure(figsize=(10,7))
    sns.heatmap(data_corr, fmt = ".2f", square = True, annot= True, linewidths= .3, mask=trimask, vmax=1, vmin=-1)
    plt.title('Correlation matrix')
    plt.tight_layout()
    plt.show()


def plot_target_corr(data : pandas.DataFrame, column: str) -> None:
    """Create correlation plot between numerical variables and target variable in a given dataframe.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame
    column : str
        target variable


    Returns
    -------
    None
        correlation plot with target variable
    
    """
    data_corr = data.corr(numeric_only=True)
    data_corr_y = data_corr[[column]].drop(column, axis=0).sort_values(by=column, ascending=False)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(t='Correlation with target \n', x=0.35, y=1)
    sns.heatmap(data_corr_y, fmt = ".2f", square = True, annot= True, linewidths= .3, ax=ax1, vmax=1, vmin=-1)
    data.corr(numeric_only=True, method= "pearson")[column]\
    .drop(column, axis=0)\
    .sort_values(ascending=True)\
    .plot(figsize=(15,5), kind="barh", colormap='RdYlBu', ax=ax2)
    plt.tight_layout()
    plt.show()


def multiple_plots(data: pandas.DataFrame, columns: list, nrows: int, ncols: int, kind: str, target=None) -> None:
    """Create multiple plots in a grid.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame
    columns : list
        list of DataFrame columns
    nrows : int
        number of rows in the figure
    ncols : int
        number of columns in the figure
    kind : str
        'boxplot' or 'countplot' or 'histplot' or 'scatterplot'
    target : str
        target variable


    Returns
    -------
    None
        multiple plots
    
    """             
    fig = plt.figure(figsize=((ncols*5), (nrows*5)))
    axes = fig.subplots(nrows=nrows, ncols=ncols)
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    for ax, column in zip(axes.flat, columns):
        ax.set_title(f'{column} distribution')
        if kind == 'boxplot' and target != None:
            sns.boxplot(x=data[column], ax=ax, color='steelblue', y=data[target])
        if kind == 'boxplot' and target == None:
            sns.boxplot(x=data[column], ax=ax, color='steelblue')
        if kind == 'countplot'and target == None:
            sns.countplot(x=data[column], ax=ax, color='steelblue')
        #if kind == 'countplot' and target != None:
            #sns.countplot(x=data[column], ax=ax, color='steelblue', hue=data[target])
        if kind == 'histplot':
            sns.histplot(x=data[column], kde=True, ax=ax, color='steelblue')
        if kind == 'scatterplot':
            sns.scatterplot(x=data[column], y=data[target], ax=ax, color='steelblue')
            ax.set_title(f"{target} ~ {column}")
    
    counter = 0
    counter += (nrows*ncols - len(columns))
    while counter > 0:
        axes.flat[-counter].axis('off')
        counter += -1
    plt.tight_layout()
    plt.show()
