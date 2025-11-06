import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']
    volume = df['volume']
    amount = df['amount']
    
    # Intraday momentum reversal (previous day's momentum tends to reverse)
    prev_ret = close.pct_change(1)
    momentum_reversal = -prev_ret
    
    # Volume-weighted volatility breakout (high volatility with high volume suggests trend continuation)
    true_range = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    vol_weighted_vol = (true_range * volume) / amount
    vol_breakout = vol_weighted_vol.rolling(5).mean() / vol_weighted_vol.rolling(20).mean()
    
    # Overnight gap mean reversion (large gaps tend to revert during the day)
    overnight_gap = (open_ - close.shift(1)) / close.shift(1)
    gap_reversion = -overnight_gap.rolling(10).apply(lambda x: x[-1] / (x.std() + 1e-8))
    
    # Combine components with equal weighting
    heuristics_matrix = momentum_reversal + vol_breakout + gap_reversion
    
    return heuristics_matrix
