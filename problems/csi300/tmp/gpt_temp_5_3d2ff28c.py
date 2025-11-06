import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    returns = close.pct_change()
    volatility = (high - low) / close
    
    momentum_short = close.pct_change(5)
    momentum_long = close.pct_change(20)
    momentum_ratio = momentum_short / momentum_long
    
    volume_ma = volume.rolling(10).mean()
    volume_ratio = volume / volume_ma
    
    vwap = amount / volume
    price_vwap_ratio = close / vwap
    
    volatility_weight = 1 / (1 + volatility.rolling(10).mean())
    
    reversal_signal = -returns.rolling(3).mean() * volume_ratio
    
    factor = (
        momentum_ratio * 0.4 +
        price_vwap_ratio.pct_change(5) * 0.3 +
        reversal_signal * volatility_weight * 0.3
    )
    
    heuristics_matrix = factor
    return heuristics_matrix
