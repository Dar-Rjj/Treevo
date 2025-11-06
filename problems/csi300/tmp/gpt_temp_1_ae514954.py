import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Short-term reversal component (3-day)
    ret_3d = close.pct_change(3)
    rev_factor = -ret_3d.rolling(window=10).rank(pct=True)
    
    # Medium-term momentum component (21-day)
    mom_21d = close.pct_change(21)
    vol_adjusted_mom = mom_21d / volume.rolling(window=21).std()
    mom_factor = vol_adjusted_mom.rolling(window=20).apply(
        lambda x: (x[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Dynamic threshold crossover
    short_ma = close.rolling(window=8).mean()
    long_ma = close.rolling(window=21).mean()
    ma_ratio = (short_ma / long_ma - 1).rolling(window=15).skew()
    
    # Combine components with residual emphasis
    raw_factor = rev_factor * 0.4 + mom_factor * 0.6
    residual = raw_factor - raw_factor.rolling(window=30).mean()
    
    heuristics_matrix = raw_factor + ma_ratio * residual
    return heuristics_matrix
