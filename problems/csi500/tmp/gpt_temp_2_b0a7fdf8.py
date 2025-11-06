import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Short-term acceleration momentum (3-day ROC of 5-day ROC)
    roc_5 = close.pct_change(5)
    momentum_accel = roc_5.pct_change(3)
    
    # Long-term mean reversion (21-day price to moving average ratio)
    ma_21 = close.rolling(21).mean()
    price_to_ma = close / ma_21 - 1
    
    # Dynamic volatility-adjusted reversal threshold
    vol_10 = close.pct_change().rolling(10).std()
    threshold = vol_10 * 2.0
    
    # Residual momentum after accounting for mean reversion
    residual = momentum_accel - (price_to_ma * 0.5)
    
    # Volume-confirmed signals
    volume_rank = volume.rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    volume_weight = np.where(volume_rank > 0.7, 1.2, 1.0)
    
    # Combined factor with dynamic thresholding
    heuristics_matrix = residual * volume_weight * np.where(np.abs(residual) > threshold, 1.5, 0.8)
    
    return heuristics_matrix
