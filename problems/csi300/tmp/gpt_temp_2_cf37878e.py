import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    # Price range efficiency (normalized daily range)
    range_eff = (close - low) / (high - low + 1e-8)
    
    # Volume-weighted price trend
    vwap = amount / (volume + 1e-8)
    price_trend = close / vwap - 1
    
    # Volatility measure (rolling 20-day)
    vol = close.pct_change().rolling(20).std()
    
    # Momentum with volatility adjustment
    mom_short = close.pct_change(5)
    mom_long = close.pct_change(20)
    vol_adj_mom = mom_short / (vol + 1e-8) - mom_long / (vol + 1e-8)
    
    # Mean reversion component for low volatility periods
    low_vol_mask = vol < vol.rolling(60).median()
    mean_rev = -close.pct_change(3) * low_vol_mask
    
    # Combine components
    heuristics_matrix = range_eff * 0.3 + price_trend * 0.4 + vol_adj_mom * 0.2 + mean_rev * 0.1
    
    return heuristics_matrix
