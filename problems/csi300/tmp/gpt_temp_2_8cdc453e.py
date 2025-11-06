import pandas as pd
import numpy as np

def heuristics_v2(df):
    open_ = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Volatility clustering regime detection
    gap_magnitude = (open_ - close.shift(1)).abs() / close.shift(1)
    vol_cluster = gap_magnitude.rolling(8).std() / (gap_magnitude.rolling(16).std() + 1e-8)
    
    # Conditional autocorrelation breakpoints
    ret_3 = close.pct_change(3)
    autocorr_break = ret_3.rolling(6).corr(ret_3.shift(2)) - ret_3.rolling(12).corr(ret_3.shift(4))
    
    # Entropy-weighted volume divergence
    vol_entropy = -volume.rolling(5).apply(lambda x: (x/x.sum() * np.log(x/x.sum() + 1e-8)).sum())
    vol_trend = volume.rolling(8).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    vol_divergence = vol_entropy * np.sign(vol_trend)
    
    # Regime-sensitive combination
    regime_signal = np.tanh(vol_cluster * 2) * autocorr_break
    heuristics_matrix = regime_signal * (1 + np.abs(vol_divergence))
    
    return heuristics_matrix
