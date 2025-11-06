import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    vwap = amount / volume.replace(0, np.nan)
    
    price_range = (high - low).replace(0, np.nan)
    normalized_close = (close - low) / price_range
    
    volume_rank = volume.rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    momentum_5 = close.pct_change(5)
    momentum_10 = close.pct_change(10)
    momentum_divergence = momentum_5 - momentum_10
    
    reversal_strength = (close - close.rolling(5).mean()) / close.rolling(5).std()
    
    vwap_deviation = (close - vwap) / vwap
    
    volume_weighted_reversal = reversal_strength * volume_rank
    
    alpha = (normalized_close * 0.3 + 
             momentum_divergence * 0.25 + 
             volume_weighted_reversal * 0.25 + 
             vwap_deviation * 0.2)
    
    heuristics_matrix = alpha
    
    return heuristics_matrix
