import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame, sector_trends: pd.Series) -> pd.Series:
    # Calculate the Volume Weighted Average Price (VWAP) with a dynamic window
    vwap = (df['amount'] / df['volume']).rolling(window=df['volume'].rolling(window=10).mean().astype(int)).mean()
    
    # Calculate the Exponentially Weighted Moving Average (EWMA) of the close price with a dynamic span
    ewma_close = df['close'].ewm(span=df['volume'].rolling(window=10).mean().astype(int), adjust=False).mean()
    
    # Calculate the log returns
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Momentum factor: 60-day moving average return
    momentum_60d = df['close'].pct_change(periods=60)
    
    # Volatility factor: 30-day standard deviation of log returns
    volatility_30d = log_returns.rolling(window=30).std()
    
    # Liquidity factor: 10-day mean of volume
    liquidity_10d = df['volume'].rolling(window=10).mean()
    
    # Trend strength: 45-day rolling correlation between VWAP and EWMA
    trend_strength = vwap.rolling(window=45).corr(ewma_close)
    
    # Market microstructure: 15-day mean of the difference between high and low prices
    market_microstructure = (df['high'] - df['low']).rolling(window=15).mean()
    
    # Sector/industry trends
    sector_factor = sector_trends.reindex(df.index, method='ffill')
    
    # Adaptive weight adjustments using machine learning
    # For simplicity, we use a linear combination of factors
    weights = np.array([0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1])
    factors = pd.concat([momentum_60d, volatility_30d, liquidity_10d, trend_strength, market_microstructure, vwap, ewma_close], axis=1).fillna(0)
    factor_values = (factors * weights).sum(axis=1)
    
    # Generate alpha factor as a weighted combination of the above metrics
    factor = (factor_values / (volatility_30d + 1e-7)) * (vwap / (ewma_close + 1e-7)) * (liquidity_10d / (df['volume'] + 1e-7)) * trend_strength * (market_microstructure + 1) * sector_factor
    return factor
