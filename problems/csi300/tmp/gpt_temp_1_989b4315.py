import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate log returns for stability
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate the 30-day Exponentially Weighted Moving Average (EWMA) of close prices
    ewma_30d = df['close'].ewm(span=30, adjust=False).mean()
    
    # Compute Volume-Weighted Average Price (VWAP)
    vwap = (df['volume'] * df[['open', 'high', 'low', 'close']].mean(axis=1)).cumsum() / df['volume'].cumsum()
    
    # Calculate the 5-day standard deviation of the volume to measure volatility
    volume_5d_std = df['volume'].rolling(window=5).std()
    
    # Calculate the 60-day and 10-day Exponentially Weighted Moving Standard Deviation (EWMStd) of log returns
    ewmstd_60d = log_returns.ewm(span=60, adjust=False).std()
    ewmstd_10d = log_returns.ewm(span=10, adjust=False).std()
    
    # Calculate the 200-day Exponentially Weighted Moving Average (EWMA) of close prices for long-term trends
    ewma_200d = df['close'].ewm(span=200, adjust=False).mean()
    
    # Adaptive weights for mean reversion and momentum
    adaptive_weight_mr = 1 / (ewmstd_60d + 1e-8)
    adaptive_weight_mm = 1 / (ewmstd_10d + 1e-8)
    
    # Additional multi-period volatility measures
    ewmstd_30d = log_returns.ewm(span=30, adjust=False).std()
    ewmstd_90d = log_returns.ewm(span=90, adjust=False).std()
    
    # Composite adaptive weight incorporating multi-period volatility
    composite_adaptive_weight = (adaptive_weight_mr + adaptive_weight_mm) / (ewmstd_30d + ewmstd_90d + 1e-8)
    
    # Integrate sector trends by calculating the 30-day EWMA of the ratio of the stock's close price to the sector's average close price
    sector_ewma_30d = (df['close'] / df.groupby('sector')['close'].transform('mean')).ewm(span=30, adjust=False).mean()
    
    # Generate alpha factor as a combination of log returns, EWMA, VWAP, volume volatility, EWMStd, and sector trend
    alpha_factor = (
        (log_returns + (df['close'] - ewma_30d) / df['close']) * 
        (vwap / df['close']) * 
        (volume_5d_std / df['volume']) * 
        composite_adaptive_weight * 
        (ewma_200d / df['close']) * 
        sector_ewma_30d
    )
    
    return alpha_factor
