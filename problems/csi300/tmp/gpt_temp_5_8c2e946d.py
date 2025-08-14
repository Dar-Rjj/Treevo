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
    smoothed_combined_factor = combined_factor.ewm(span=14).mean()
    
    # Apply Volume Weighting
    volume_weighted_factor = smoothed_combined_factor * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    prev_day_close_gap = df['open'] - df['close'].shift(1)
    volume_weighted_smoothed_factor_with_gap = volume_weighted_factor + prev_day_close_gap
    
    # Integrate Long-Term Momentum
    long_term_return = df['close'] - df['close'].shift(50)
    normalized_long_term_return = long_term_return / high_low_range
    
    # Include Dynamic Volatility Component
    rolling_std = intraday_return.rolling(window=20).std()
    atr = df[['high', 'low', 'close']].apply(lambda x: np.max(x) - np.min(x), axis=1)
    atr = atr.rolling(window=14).mean()
    combined_volatility = (rolling_std + atr) / 2
    volume_adjusted_volatility = combined_volatility * df['volume']
    
    # Calculate Relative Strength
    stock_cumulative_return = df['close'] / df['close'].shift(20)
    reference_index_cumulative_return = df['reference_index_close'] / df['reference_index_close'].shift(20)
    relative_strength = stock_cumulative_return / reference_index_cumulative_return
    
    # Final Factor Calculation
    final_factor = (volume_weighted_smoothed_factor_with_gap +
                    normalized_long_term_return +
                    volume_adjusted_volatility +
                    relative_strength)
    
    # Apply Non-Linear Transformation
    final_factor = np.log(1 + final_factor)
    
    return final_factor
