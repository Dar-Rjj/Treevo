import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    def hurst_exponent(ts, window=20):
        lags = range(2, 8)
        tau = [np.std(np.subtract(ts[lag:].values, ts[:-lag].values)) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    
    hurst_rolling = close.rolling(30).apply(hurst_exponent, raw=False)
    
    volume_flow = (volume * (close - (high + low) / 2)).rolling(12).sum()
    volatility_regime = close.pct_change().rolling(8).std() / close.pct_change().rolling(25).std()
    adjusted_flow = volume_flow / (volatility_regime + 1e-8)
    
    momentum_confirmation = (close.rolling(5).mean() / close.rolling(15).mean() - 1) * (close.rolling(8).std() / close.rolling(20).std())
    
    heuristics_matrix = hurst_rolling * adjusted_flow * np.sign(momentum_confirmation)
    
    return heuristics_matrix
