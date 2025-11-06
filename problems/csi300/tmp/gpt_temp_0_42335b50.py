import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']
    volume = df['volume']
    
    # Volatility-normalized intraday momentum
    intraday_return = (close - open_) / open_
    volatility = (high - low).rolling(window=12).std()
    normalized_momentum = intraday_return / (volatility + 1e-8)
    
    # Gap-fill probability
    overnight_gap = (open_ - close.shift(1)) / close.shift(1)
    gap_fill_probability = -np.sign(overnight_gap) * (high.rolling(window=8).max() - low.rolling(window=8).min()) / close.rolling(window=8).mean()
    
    # Volume persistence
    volume_trend = volume.rolling(window=6).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
    volume_persistence = volume_trend * volume / volume.rolling(window=10).mean()
    
    # Combined alpha factor
    heuristics_matrix = normalized_momentum * gap_fill_probability * volume_persistence
    
    return heuristics_matrix
