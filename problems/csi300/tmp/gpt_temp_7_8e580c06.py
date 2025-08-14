import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.optimize import minimize

def evaluate_performance(factor, returns):
    """Evaluate the performance of a factor using Sharpe Ratio."""
    excess_returns = returns - factor.mean()
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    return -sharpe_ratio  # Minimize negative Sharpe Ratio

def optimal_window(df, window_range, metric_func):
    """Calculate the optimal window size for a given metric function."""
    best_sharpe = float('-inf')
    best_window = None
    for window in window_range:
        factor = metric_func(df, window)
        returns = df['close'].pct_change().shift(-1).dropna()
        factor = factor.reindex(returns.index).dropna()
        sharpe = -evaluate_performance(factor, returns)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_window = window
    return best_window

def logarithmic_returns(df, window):
    """Calculate the sum of logarithmic returns over N days."""
    log_returns = np.log(df['close']).diff()
    return log_returns.rolling(window=window).sum()

def volume_acceleration(df, window):
    """Calculate the sum of volume changes over M days."""
    volume_changes = df['volume'].pct_change()
    return volume_changes.rolling(window=window).sum()

def volatility(df, window):
    """Calculate the rolling standard deviation of daily logarithmic returns."""
    log_returns = np.log(df['close']).diff()
    return log_returns.rolling(window=window).std()

def wma(df, window):
    """Calculate the Weighted Moving Average (WMA) for close prices."""
    weights = np.arange(1, window + 1)
    wma = df['close'].rolling(window=window).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    return wma

def high_low_spread(df, window):
    """Calculate the average high-low spread over R days."""
    spread = df['high'] - df['low']
    return spread.rolling(window=window).mean()

def heuristics_v2(df):
    # Calculate optimal windows
    n_days = optimal_window(df, range(5, 30), logarithmic_returns)
    m_days = optimal_window(df, range(5, 30), volume_acceleration)
    p_days = optimal_window(df, range(5, 30), volatility)
    q_days = optimal_window(df, range(5, 30), wma)
    r_days = optimal_window(df, range(5, 30), high_low_spread)

    # Calculate price momentum
    price_momentum = logarithmic_returns(df, n_days)
    
    # Calculate volume acceleration
    volume_accel = volume_acceleration(df, m_days)
    
    # Calculate volatility
    vol = volatility(df, p_days)
    adjusted_price_momentum = price_momentum / vol
    
    # Calculate WMA and trend indicator
    wma_series = wma(df, q_days)
    trend_indicator = (df['close'] > wma_series).astype(int) * 2 - 1  # +1 if close > WMA, -1 otherwise
    
    # Calculate high-low spread
    avg_high_low_spread = high_low_spread(df, r_days)
    median_high_low_spread = (df['high'] - df['low']).median()
    high_low_ratio = avg_high_low_spread / median_high_low_spread
    adjusted_volume_accel = volume_accel * high_low_ratio
    
    # Combine factors
    alpha_factor = adjusted_price_momentum * adjusted_volume_accel + trend_indicator
    
    return alpha_factor
