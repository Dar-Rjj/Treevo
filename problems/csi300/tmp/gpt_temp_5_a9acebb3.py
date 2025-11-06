import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Volatility Regime Component
    # Calculate Intraday Volatility Ratio
    high_low_range = data['high'] - data['low']
    close_open_gap = np.abs(data['close'] - data['open'])
    # Avoid division by zero by replacing zeros with a small value
    close_open_gap = close_open_gap.replace(0, 1e-8)
    volatility_ratio = high_low_range / close_open_gap
    
    # Calculate Volatility Persistence (Autocorrelation over 8 days)
    volatility_persistence = volatility_ratio.rolling(window=8, min_periods=8).apply(
        lambda x: x.autocorr(lag=1), raw=False
    )
    
    # Liquidity Efficiency Component
    # Calculate Volume-to-Amount Efficiency
    volume_amount_efficiency = data['volume'] / data['amount']
    # Handle potential division by zero
    volume_amount_efficiency = volume_amount_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Calculate Liquidity Momentum (Linear regression slope over 6 days)
    def calc_slope(window):
        if len(window) < 6 or window.isna().any():
            return np.nan
        try:
            x = np.arange(len(window))
            slope, _, _, _, _ = linregress(x, window.values)
            return slope
        except:
            return np.nan
    
    liquidity_momentum = volume_amount_efficiency.rolling(window=6, min_periods=6).apply(
        calc_slope, raw=False
    )
    
    # Combine components to create final alpha factor
    # Normalize components to make them comparable
    volatility_persistence_norm = (volatility_persistence - volatility_persistence.rolling(window=20, min_periods=1).mean()) / volatility_persistence.rolling(window=20, min_periods=1).std()
    liquidity_momentum_norm = (liquidity_momentum - liquidity_momentum.rolling(window=20, min_periods=1).mean()) / liquidity_momentum.rolling(window=20, min_periods=1).std()
    
    # Final alpha factor: Volatility-Adjusted Liquidity Momentum
    alpha_factor = volatility_persistence_norm * liquidity_momentum_norm
    
    return alpha_factor
