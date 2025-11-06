import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']
    volume = df['volume']
    amount = df['amount']
    
    intraday_momentum = (close - open_) / (high - low + 1e-12)
    volume_trend = volume / volume.rolling(window=20).mean()
    volatility_breakout = (high - low) / (high.rolling(window=20).mean() - low.rolling(window=20).mean() + 1e-12)
    
    vwap = amount / (volume + 1e-12)
    price_vwap_deviation = (close - vwap) / vwap
    
    factor = intraday_momentum * volume_trend * volatility_breakout * price_vwap_deviation
    factor = factor.replace([np.inf, -np.inf], np.nan)
    
    heuristics_matrix = factor
    return heuristics_matrix
