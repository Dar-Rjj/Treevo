import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    vwap = amount / volume.replace(0, np.nan)
    
    price_range = (high - low) / close.shift(1)
    volume_rank = volume.rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    divergence = (close - vwap) / close * volume_rank
    
    volatility = close.pct_change().rolling(20).std()
    momentum = (close / close.shift(5) - 1) / volatility.replace(0, np.nan)
    
    mean_reversion = (close - close.rolling(10).mean()) / close.rolling(10).std().replace(0, np.nan)
    
    heuristics_matrix = divergence * 0.6 + momentum * 0.3 - mean_reversion * 0.1
    
    return heuristics_matrix
