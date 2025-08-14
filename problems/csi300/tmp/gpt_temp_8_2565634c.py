import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, median_absolute_deviation

def heuristics_v2(df, N=20, M=20, momentum_threshold=1.5, volume_weight=0.3, volatilities_weight=0.7):
    # Calculate Log Returns
    close_prices = df['close']
    log_returns = np.log(close_prices).diff().dropna()
    
    # Compute Momentum
    cumulative_log_returns = log_returns.rolling(window=N).sum()
    upper_threshold = cumulative_log_returns.mean() + momentum_threshold * cumulative_log_returns.std()
    lower_threshold = cumulative_log_returns.mean() - momentum_threshold * cumulative_log_returns.std()
    clipped_cumulative_log_returns = np.clip(cumulative_log_returns, lower_threshold, upper_threshold)
    
    # Adjust for Volume
    volumes = df['volume']
    volume_relative_to_mean = (volumes / volumes.rolling(window=M).mean()).fillna(0)
    volume_adjusted_momentum = clipped_cumulative_log_returns * volume_relative_to_mean
    
    # Determine Absolute Price Changes
    absolute_price_changes = close_prices.diff().abs().fillna(0)
    
    # Calculate Advanced Volatility Measures
    std_volatility = absolute_price_changes.rolling(window=M).std().fillna(0)
    ema_volatility = absolute_price_changes.ewm(span=M, adjust=False).mean().fillna(0)
    iqr_volatility = absolute_price_changes.rolling(window=M).quantile(0.75) - absolute_price_changes.rolling(window=M).quantile(0.25)
    mad_volatility = median_absolute_deviation(absolute_price_changes, scale='normal', axis=0, nan_policy='omit')
    kurtosis_volatility = kurtosis(absolute_price_changes, nan_policy='omit')
    
    # Final Factor Calculation
    total_volatility = (std_volatility + ema_volatility + iqr_volatility + mad_volatility + kurtosis_volatility) / 5
    factor_values = (volume_weight * volume_adjusted_momentum + volatilities_weight * total_volatility)
    
    return factor_values
