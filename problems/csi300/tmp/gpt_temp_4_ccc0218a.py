import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Short-term momentum acceleration
    ret_5 = close.pct_change(5)
    ret_10 = close.pct_change(10)
    momentum_accel = ret_5 - ret_10
    
    # Volatility-adjusted trend strength
    atr = (high - low).rolling(20).mean()
    vol_adj_trend = (close - close.rolling(50).mean()) / atr
    
    # Regime detection using crossover
    short_ma = close.rolling(10).mean()
    long_ma = close.rolling(30).mean()
    regime = (short_ma > long_ma).astype(int)
    
    # Factor combination based on regime
    heuristics_matrix = momentum_accel * vol_adj_trend * regime - momentum_accel * vol_adj_trend * (1 - regime)
    
    return heuristics_matrix
