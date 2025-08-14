import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate Volume Weighted Average Price (VWAP)
    vwap = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
    
    # Calculate Exponentially Weighted Moving Average (EWMA) of VWAP with a span of 10 days
    ewma_vwap = vwap.ewm(span=10, adjust=False).mean()
    
    # Calculate log returns of the close price
    log_returns = np.log(df['close']).diff()
    
    # Calculate the long-term trend by taking the 60-day moving average of the log returns
    lt_trend = log_returns.rolling(window=60).mean()
    
    # Calculate robust volatility as the 30-day standard deviation of log returns
    robust_volatility = log_returns.rolling(window=30).std()
    
    # Incorporate momentum by calculating the 20-day rate of change in close price
    momentum = df['close'].pct_change(periods=20)
    
    # Incorporate price-volume interaction by calculating the 10-day correlation between log returns and volume
    pv_interaction = df[['close', 'volume']].apply(lambda x: np.log(x)).rolling(window=10).corr().iloc[::2, -1].reset_index(drop=True)
    
    # Generate alpha factor as a combination of the above metrics
    factor = (ewma_vwap + lt_trend + robust_volatility + momentum + pv_interaction) / 5
    
    return factor
