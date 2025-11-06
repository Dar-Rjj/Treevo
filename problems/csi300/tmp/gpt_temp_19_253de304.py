import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Short-term price reversal with momentum crossover: 2-day vs 5-day momentum difference
    momentum_2d = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_crossover = momentum_2d - momentum_5d
    
    # Volume-pressure indicator: ratio of current volume to 3-day average volume
    volume_3d_avg = df['volume'].rolling(window=3, min_periods=2).mean()
    volume_pressure = df['volume'] / (volume_3d_avg + 1e-7)
    
    # Intraday strength persistence: 3-day rolling correlation between daily range position and next-day return
    daily_range_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    next_day_return = df['close'].pct_change().shift(-1)
    
    # Calculate rolling correlation over 3-day window
    range_return_corr = pd.Series(index=df.index, dtype=float)
    for i in range(2, len(df)):
        if i >= 2:
            window_range = daily_range_position.iloc[i-2:i+1]
            window_return = next_day_return.iloc[i-2:i+1]
            if not window_range.isna().any() and not window_return.isna().any():
                range_return_corr.iloc[i] = window_range.corr(window_return)
    
    # Adaptive volatility measure: 3-day rolling standard deviation of 1-hour equivalent returns (using 6.5 trading hours)
    intraday_volatility = df['close'].pct_change().rolling(window=3, min_periods=2).std() * np.sqrt(6.5)
    
    # Signal interaction: momentum crossover amplified by volume pressure and range-return correlation
    raw_factor = momentum_crossover * volume_pressure * (1 + range_return_corr.fillna(0))
    
    # Adaptive volatility normalization using intraday volatility measure
    factor = raw_factor / (intraday_volatility + 1e-7)
    
    return factor
