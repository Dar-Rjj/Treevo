import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Calculate Intraday High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Combine Intraday Return and High-Low Range
    combined_factor = intraday_return * high_low_range
    smoothed_factor = combined_factor.ewm(span=14).mean()
    
    # Apply Volume Weighting
    volume_weighted_factor = smoothed_factor * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    prev_day_close_gap = df['open'].diff()
    gap_factor = volume_weighted_factor + prev_day_close_gap
    
    # Integrate Long-Term Momentum
    long_term_return = df['close'] - df['close'].shift(50)
    normalized_long_term_return = long_term_return / high_low_range
    
    # Include Enhanced Dynamic Volatility Component
    rolling_std = intraday_return.rolling(window=20).std()
    atr = df[['high', 'low', 'close']].apply(lambda x: np.max(np.abs(x)) - np.min(np.abs(x)), axis=1)
    atr = atr.rolling(window=14).mean()
    combined_volatility = (rolling_std + atr) / 2
    
    # Adjust Volatility Component with Volume
    volume_adjusted_volatility = combined_volatility * df['volume']
    
    # Final Factor Calculation
    final_factor = gap_factor + normalized_long_term_return + volume_adjusted_volatility
    final_factor = np.log(1 + final_factor)
    
    return final_factor
