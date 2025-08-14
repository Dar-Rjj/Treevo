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
    
    # Calculate the 14-day Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Incorporate liquidity factor
    liquidity = df['amount'] / df['volume']
    
    # Market psychology: Calculate the 20-day Average True Range (ATR)
    tr = df[['high', 'low', 'close']].apply(lambda x: np.max(x) - np.min(x), axis=1)
    atr_20d = tr.rolling(window=20).mean()
    
    # Generate alpha factor as a combination of log returns, EWMA, VWAP, volume volatility, EWMStd, RSI, liquidity, and ATR
    alpha_factor = (
        (log_returns + (df['close'] - ewma_30d) / df['close']) * 
        (vwap / df['close']) * 
        (volume_5d_std / df['volume']) * 
        composite_adaptive_weight * 
        (ewma_200d / df['close']) * 
        (rsi / 50) * 
        (liquidity / df['close']) * 
        (atr_20d / df['close'])
    )
    
    # Alternative volatility measure: Parkinson's High-Low Volatility
    high_low_volatility = np.sqrt((np.log(df['high'] / df['low']))**2 / (4 * np.log(2)))
    
    # Combine alternative volatility with existing factors
    alpha_factor = alpha_factor * (1 + high_low_volatility)
    
    return alpha_factor
