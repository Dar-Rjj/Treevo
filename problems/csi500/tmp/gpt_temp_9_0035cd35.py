import pandas as pd
import numpy as np

def heuristics_v2(df):
    open_ = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    price_range = (high - low) / close
    volume_trend = volume.rolling(5).mean() / volume.rolling(20).mean()
    
    reversal_signal = -(close - close.shift(3)) / close.shift(3)
    volatility_adjustment = price_range.rolling(10).std()
    
    liquidity_momentum = volume_trend * (close / close.rolling(5).mean() - 1)
    cross_sectional_rank = close.rolling(10).apply(lambda x: (x[-1] - x.mean()) / x.std())
    
    heuristics_matrix = reversal_signal / (volatility_adjustment + 1e-8) + liquidity_momentum + cross_sectional_rank
    heuristics_matrix = heuristics_matrix.replace([float('inf'), -float('inf')], float('nan'))
    
    return heuristics_matrix
