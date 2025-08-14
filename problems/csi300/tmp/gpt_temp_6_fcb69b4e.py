import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame, sector_data: pd.Series) -> pd.Series:
    # Calculate log returns for stability
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate the 30-day Exponentially Weighted Moving Average (EWMA) of close prices
    ewma_30d = df['close'].ewm(span=30, adjust=False).mean()
    
    # Compute Volume-Weighted Average Price (VWAP)
    vwap = (df['volume'] * df[['open', 'high', 'low', 'close']].mean(axis=1)).cumsum() / df['volume'].cumsum()
    
    # Adaptive volatility adjustment factor
    adaptive_volatility = df['volume'].rolling(window=10).std() / df['volume']
    
    # Calculate the 60-day Exponentially Weighted Moving Standard Deviation (EWMStd) of log returns for mean reversion
    ewmstd_60d = log_returns.ewm(span=60, adjust=False).std()
    
    # Calculate the 120-day Exponentially Weighted Moving Standard Deviation (EWMStd) of log returns for longer-term volatility
    ewmstd_120d = log_returns.ewm(span=120, adjust=False).std()
    
    # Sector-specific sentiment indicator
    sector_sentiment = sector_data / sector_data.shift(1)
    
    # Generate alpha factor as a combination of log returns, EWMA, VWAP, adaptive volatility, EWMStd, and sector sentiment
    alpha_factor = (log_returns + (df['close'] - ewma_30d) / df['close']) * (vwap / df['close']) * adaptive_volatility * (ewmstd_120d / (ewmstd_60d + 1e-8)) * sector_sentiment
    
    return alpha_factor
