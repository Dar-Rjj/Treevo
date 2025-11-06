import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    intraday_high_deviation = (high - close) / close
    intraday_low_deviation = (close - low) / close
    mean_reversion_signal = intraday_low_deviation - intraday_high_deviation
    
    volume_persistence = volume.rolling(10).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 10 and np.std(x) > 0 else 0)
    volatility_regime = close.pct_change().rolling(20).std()
    
    volume_weight = 1 / (1 + np.exp(-volume_persistence * 5))
    volatility_adjustment = 1 / (volatility_regime + 1e-8)
    
    heuristics_matrix = mean_reversion_signal * volume_weight * volatility_adjustment
    heuristics_matrix = heuristics_matrix.rename('heuristics_v2')
    
    return heuristics_matrix
