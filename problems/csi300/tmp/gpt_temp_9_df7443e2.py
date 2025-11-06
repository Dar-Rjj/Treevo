import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum acceleration factor with volume confirmation
    # Combines price momentum acceleration with volume trend divergence
    # Positive values indicate strong upward momentum with increasing volume
    
    # Calculate short-term and medium-term momentum
    mom_short = df['close'] - df['close'].shift(5)
    mom_medium = df['close'] - df['close'].shift(10)
    
    # Calculate momentum acceleration (rate of change in momentum)
    mom_acceleration = mom_short - mom_medium.shift(5)
    
    # Calculate volume trend using rolling regression slope
    volume_trend = df['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    
    # Calculate price range efficiency (how much of the daily range is utilized)
    range_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    
    # Combine factors with emphasis on momentum acceleration and volume confirmation
    factor = (mom_acceleration * volume_trend * range_efficiency.abs())
    
    return factor
