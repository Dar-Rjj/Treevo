import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Extreme volume momentum with volatility-filtered efficiency: Combines extreme volume spikes 
    (90th percentile threshold) with long-term momentum (60-day), normalized by volatility (30-day ATR). 
    Filters using efficiency ratios (15-day rolling) to identify clean trend signals.
    
    Factor interpretation:
    - Extreme volume identification: Uses 90th percentile threshold over 60-day window to detect 
      significant volume events that often precede trend changes
    - Long-term momentum: 60-day price momentum captures sustained directional moves
    - Volatility normalization: 30-day average true range provides regime-appropriate scaling
    - Efficiency filtering: 15-day rolling efficiency ratio (absolute price change / daily range)
      ensures price movement quality and reduces noise
    - Economic rationale: Extreme volume spikes coupled with long-term momentum in efficient 
      price discovery environments signal high-confidence trend continuation
    """
    
    # Extreme volume spike detection using 90th percentile threshold over 60 days
    volume_60d_percentile = df['volume'].rolling(window=60).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 90)) if len(x) == 60 else np.nan, 
        raw=False
    )
    
    # Long-term momentum (60-day)
    momentum_60d = (df['close'] - df['close'].shift(60)) / df['close'].shift(60)
    
    # Volatility measurement using 30-day average true range
    volatility_30d = ((df['high'] - df['low']).rolling(window=30).mean())
    
    # Efficiency ratio: 15-day rolling average of daily price efficiency
    daily_efficiency = abs(df['close'] - df['close'].shift(1)) / ((df['high'] - df['low']) + 1e-7)
    efficiency_15d = daily_efficiency.rolling(window=15).mean()
    
    # Combine components: extreme volume * momentum, normalized by volatility, filtered by efficiency
    volume_momentum = volume_60d_percentile * momentum_60d
    volatility_normalized = volume_momentum / (volatility_30d + 1e-7)
    alpha_factor = volatility_normalized * efficiency_15d
    
    return alpha_factor
