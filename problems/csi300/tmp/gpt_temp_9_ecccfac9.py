import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Price trend acceleration
    short_ma = close.rolling(window=5).mean()
    long_ma = close.rolling(window=20).mean()
    trend_acceleration = (short_ma - long_ma) / long_ma
    
    # Relative strength rotation
    price_rank = close.rolling(window=10).apply(lambda x: x.rank().iloc[-1] / len(x))
    high_low_range = (high.rolling(window=5).max() - low.rolling(window=5).min()) / close
    relative_strength = price_rank * high_low_range
    
    # Liquidity-driven momentum
    volume_trend = volume.rolling(window=10).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
    price_momentum = (close / close.shift(5) - 1)
    liquidity_momentum = volume_trend * price_momentum
    
    # Combined alpha factor
    heuristics_matrix = trend_acceleration + relative_strength + liquidity_momentum
    
    return heuristics_matrix
