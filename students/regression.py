"""
Linear regression functions for predicting cholesterol using ElasticNet.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from itertools import combinations


def train_elasticnet_grid(X_train: pd.DataFrame|np.ndarray,
                          y_train: pd.Series|np.ndarray,
                          l1_ratios: list|np.ndarray,
                          alphas: list|np.ndarray) -> pd.DataFrame:
    """
    Train ElasticNet models over a grid of hyperparameters.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training feature matrix
    y_train : np.ndarray or pd.Series
        Training target vector
    l1_ratios : list or np.ndarray
        L1 ratio values to test (0 = L2 only, 1 = L1 only)
    alphas : list or np.ndarray
        Regularization strength values to test
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['l1_ratio', 'alpha', 'r2_score', 'model']
        Contains R² scores for each parameter combination on training data
    """
    # TODO: Implement grid search
    # - Create results list
    # - For each combination of l1_ratio and alpha:
    #   - Train ElasticNet model with max_iter=5000
    #   - Calculate R² score on training data
    #   - Store results
    # - Return DataFrame with results
    results = pd.DataFrame(colunns = ['l1_ratio', 'alpha', 'r2_score', 'model'])
    
    # Combination of all the parameters:
    for l1,alpha in combinations(l1_ratios, alphas):
        model = ElasticNet(l1_ratio=l1, alpha=alpha, max_iter=5000)
        model.fit(X_train, y_train)
        r2 = r2_score(y_train, model.predict(X_train))
        results_df = pd.concat(objs = [results,pd.DataFrame({'l1_ratio': [l1], 'alpha': [alpha], 'r2_score': [r2], 'model': [model]})], ignore_index=True)
    
    return results_df


def create_r2_heatmap(results_df: pd.DataFrame, l1_ratios: list, alphas: list, output_path=None):
    """
    Create a heatmap of R² scores across l1_ratio and alpha parameters.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from train_elasticnet_grid
    l1_ratios : list or np.ndarray
        L1 ratio values used in grid
    alphas : list or np.ndarray
        Alpha values used in grid
    output_path : str, optional
        Path to save figure. If None, returns figure object
        
    Returns
    -------
    matplotlib.figure.Figure
        The heatmap figure
    """
    # TODO: Implement heatmap creation
    # - Pivot results_df to create matrix with l1_ratio on x-axis, alpha on y-axis
    # - Create heatmap using seaborn
    # - Set labels: "L1 Ratio", "Alpha", "R² Score"
    # - Add colorbar
    # - Save to output_path if provided
    # - Return figure object
    pivot_df = pd.pivot_table(results_df, values=results_df.loc[:,'r2_score'].values, index=l1_ratios, columns=alphas)
    fig = sns.heatmap(pivot_df, annot=True, cmap='coolwarm', cbar_kws={'label': r'R^2 Score'})
    
    return fig


def get_best_elasticnet_model(X_train, y_train, X_test, y_test, 
                               l1_ratios=None, alphas=None):
    """
    Find and train the best ElasticNet model on test data.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    y_train : np.ndarray or pd.Series
        Training target
    X_test : np.ndarray or pd.DataFrame
        Test features
    y_test : np.ndarray or pd.Series
        Test target
    l1_ratios : list, optional
        L1 ratio values to test. Default: [0.1, 0.3, 0.5, 0.7, 0.9]
    alphas : list, optional
        Alpha values to test. Default: [0.001, 0.01, 0.1, 1.0, 10.0]
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': fitted ElasticNet model
        - 'best_l1_ratio': best l1 ratio
        - 'best_alpha': best alpha
        - 'train_r2': R² on training data
        - 'test_r2': R² on test data
        - 'results_df': full results DataFrame
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    # TODO: Implement best model selection
    # - Train models using train_elasticnet_grid
    # - Select model with highest test R² (not training R²)
    # - Return dictionary with best model and parameters
    
    results_df = pd.DataFrame(['model','best_l1_ratio','best_alpha','train_r2','test_r2','results_df'])
    
    for l1,alpha in combinations(l1_ratios,alphas):
        model = ElasticNet(l1_ratio=l1, alpha=alpha, max_iter=5000)
        model.fit(X_train, y_train)
        r2_train = r2_score(y_train, model.predict(X_train)) # score the prediction on trained data (BAD).
        r2_test = r2_score(y_test, model.predict(X_test)) # ----- on test data (GOOD).
        results_df = pd.concat(objs = [results_df,pd.DataFrame({'l1_ratio': [l1], 
                                                                'alpha': [alpha], 
                                                                'train_r2': [r2_train], 
                                                                'test_r2': [r2_test], 
                                                                'model': [model]})], ignore_index=True)
    
    best_model = results_df.loc[results_df.loc[:,'test_r2'].idxmax(),:].to_dict(orient = 'records')
    best_model['results_df'] = results_df
    
    return best_model
