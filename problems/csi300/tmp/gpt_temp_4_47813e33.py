import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Price acceleration momentum (5-day vs 20-day ROC difference)
    roc_5 = close.pct_change(5)
    roc_20 = close.pct_change(20)
    price_accel = roc_5 - roc_20
    
    # Volatility-normalized mean reversion (using ATR)
    atr = (high - low).rolling(14).mean()
    price_deviation = (close - close.rolling(50).mean()) / atr
    
    # Volume-confirmed breakout residuals
    volume_ma = volume.rolling(20).mean()
    volume_spike = (volume - volume_ma) / volume_ma
    price_trend = close.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Dynamic threshold crossover
    upper_band = price_accel.rolling(30).quantile(0.7)
    lower_band = price_accel.rolling(30).quantile(0.3)
    momentum_signal = ((price_accel > upper_band) | (price_accel < lower_band)).astype(int)
    
    # Composite factor with interaction terms
    heuristics_matrix = (price_accel * 0.4 + 
                        price_deviation * (-0.3) + 
                        (volume_spike * price_trend) * 0.2 +
                        momentum_signal * price_accel * 0.1)
    
    return heuristics_matrix
