import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Short-term momentum acceleration (5-day ROC of 10-day ROC)
    roc_10 = close.pct_change(10)
    momentum_accel = roc_10.pct_change(5)
    
    # Long-term volatility-adjusted momentum (20-day return divided by 20-day volatility)
    returns_20 = close.pct_change(20)
    vol_20 = close.pct_change().rolling(20).std()
    vol_adjusted_momentum = returns_20 / (vol_20 + 1e-8)
    
    # Regime change detection using volume-confirmed price breakouts
    price_range = (high - low) / close
    volume_rank = volume.rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    breakout_strength = (close - close.rolling(10).mean()) / (close.rolling(10).std() + 1e-8)
    volume_confirmed = breakout_strength * volume_rank
    
    # Combine components with non-linear transformation
    factor = momentum_accel * np.tanh(vol_adjusted_momentum) * np.sign(volume_confirmed)
    
    heuristics_matrix = pd.Series(factor, index=df.index, name='heuristics_v2')
    return heuristics_matrix
