import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Volatility regime detection using multiple timeframes
    vol_short = close.pct_change().rolling(5).std()
    vol_long = close.pct_change().rolling(20).std()
    vol_regime = vol_short / vol_long
    
    # Dynamic support/resistance using rolling percentiles
    resistance = high.rolling(10).apply(lambda x: np.percentile(x, 80))
    support = low.rolling(10).apply(lambda x: np.percentile(x, 20))
    
    # Mean reversion signal from support/resistance bands
    price_position = (close - support) / (resistance - support)
    mean_reversion = 0.5 - price_position
    
    # Volume-weighted momentum with volatility adaptation
    returns_5d = close.pct_change(5)
    volume_weight = volume / volume.rolling(10).mean()
    raw_momentum = returns_5d * volume_weight
    
    # Combine signals based on volatility regime
    high_vol_signal = mean_reversion * (1.0 / (1.0 + vol_regime))
    low_vol_signal = raw_momentum * (vol_regime / (1.0 + vol_regime))
    
    heuristics_matrix = high_vol_signal + low_vol_signal
    return heuristics_matrix
