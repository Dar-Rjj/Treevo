import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 3-5 day momentum acceleration blended with volume-amount geometric alignment
    # Normalized by 10-day volatility with exponential decay for recent emphasis
    
    # 3-day momentum
    mom_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # 5-day momentum  
    mom_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Momentum acceleration (3-day vs 5-day)
    mom_accel = mom_3d - mom_5d
    
    # Volume-amount geometric alignment (geometric mean of normalized ratios)
    vol_norm = df['volume'] / df['volume'].rolling(window=5).mean()
    amt_norm = df['amount'] / df['amount'].rolling(window=5).mean()
    vol_amt_align = (vol_norm * amt_norm) ** 0.5
    
    # Blend momentum acceleration with volume-amount alignment
    raw_factor = mom_accel * vol_amt_align
    
    # 10-day volatility (using close-to-close returns)
    returns = df['close'].pct_change()
    volatility_10d = returns.rolling(window=10).std()
    
    # Normalize by volatility with small epsilon to avoid division by zero
    normalized_factor = raw_factor / (volatility_10d + 1e-7)
    
    # Apply exponential decay (half-life of 5 days) for recent data emphasis
    decay_factor = 0.5 ** (1/5)  # Daily decay rate
    weights = [decay_factor ** i for i in range(len(normalized_factor))]
    weights.reverse()
    weights = pd.Series(weights, index=normalized_factor.index)
    
    # Apply weighted average with exponential decay
    decayed_factor = normalized_factor.rolling(window=len(normalized_factor), min_periods=1).apply(
        lambda x: np.average(x, weights=weights[-len(x):]), raw=False
    )
    
    return decayed_factor
