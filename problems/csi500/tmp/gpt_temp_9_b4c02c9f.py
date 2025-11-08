import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Extract price and volume data
    close = df['close']
    volume = df['volume']
    
    # Price Momentum Component
    # Calculate multiple timeframe returns
    ret_1d = close.pct_change(1)
    ret_3d = close.pct_change(3)
    ret_5d = close.pct_change(5)
    
    # Volatility adjustment for each timeframe
    vol_5d = ret_1d.rolling(window=5).std()
    vol_10d = ret_1d.rolling(window=10).std()
    
    # Volatility-adjusted returns
    adj_ret_1d = ret_1d / vol_5d
    adj_ret_3d = ret_3d / vol_5d
    adj_ret_5d = ret_5d / vol_10d
    
    # Volume Momentum Component
    # Calculate volume trend across timeframes
    vol_ratio_short = volume / volume.shift(1)
    
    vol_avg_medium = volume.shift(1).rolling(window=5).mean()
    vol_ratio_medium = volume / vol_avg_medium
    
    vol_avg_long = volume.shift(1).rolling(window=10).mean()
    vol_ratio_long = volume / vol_avg_long
    
    # Volume volatility adjustment
    vol_returns = volume / volume.shift(1)
    vol_vol_5d = vol_returns.rolling(window=5).std()
    vol_vol_10d = vol_returns.rolling(window=10).std()
    
    # Signal Integration
    # Combine price and volume signals
    blended_short = adj_ret_1d * (vol_ratio_short / vol_vol_5d)
    blended_medium = adj_ret_3d * (vol_ratio_medium / vol_vol_5d)
    blended_long = adj_ret_5d * (vol_ratio_long / vol_vol_10d)
    
    # Weighted combination
    weights = [0.5, 0.3, 0.2]
    factor_value = (weights[0] * blended_short + 
                   weights[1] * blended_medium + 
                   weights[2] * blended_long)
    
    # Cross-Sectional Ranking
    # Rank factor value across all stocks for each date
    def cross_sectional_rank(series):
        return series.rank(pct=True)
    
    alpha_factor = factor_value.groupby(level='date').transform(cross_sectional_rank)
    
    return alpha_factor
