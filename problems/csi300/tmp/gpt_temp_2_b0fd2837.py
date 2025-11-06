import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']
    volume = df['volume']
    amount = df['amount']
    
    # Intraday momentum reversal (previous day's intraday return)
    intraday_return = (close - open_) / open_
    prev_intraday_return = intraday_return.shift(1)
    
    # Volume-weighted volatility breakout
    typical_price = (high + low + close) / 3
    volatility = typical_price.rolling(window=5).std()
    volume_weighted_vol = (volume * volatility).rolling(window=5).mean()
    vol_breakout = (typical_price - typical_price.rolling(window=5).mean()) / volume_weighted_vol
    
    # Overnight gap mean reversion
    overnight_gap = (open_ - close.shift(1)) / close.shift(1)
    gap_mean_reversion = -overnight_gap.rolling(window=10).apply(lambda x: x[-1] / (x[:-1].std() + 1e-8))
    
    # Combined alpha factor
    heuristics_matrix = prev_intraday_return - vol_breakout + gap_mean_reversion
    
    return heuristics_matrix
