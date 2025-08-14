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
    combined_factor_smoothed = combined_factor.ewm(span=14).mean()
    
    # Apply Volume Weighting
    volume_weighted_factor = combined_factor_smoothed * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    prev_day_close_gap = df['open'] - df['close'].shift(1)
    volume_weighted_factor_gap = volume_weighted_factor + prev_day_close_gap
    
    # Integrate Long-Term Momentum
    long_term_return = df['close'] - df['close'].shift(50)
    normalized_long_term_return = long_term_return / high_low_range
    
    # Include Enhanced Dynamic Volatility Component
    intraday_returns_rolling_std = intraday_return.rolling(window=20).std()
    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    combined_volatility = (intraday_returns_rolling_std + atr) / 2
    volatility_volume_adjusted = combined_volatility * df['volume']
    
    # Refine Smoothing
    double_ema = (2 * combined_factor_smoothed) - combined_factor_smoothed.ewm(span=14).mean()
    
    # Incorporate Volume-Based Features
    volume_momentum = (df['volume'] / df['volume'].ewm(span=14).mean()) - 1
    
    # Final Factor Calculation
    final_factor = (double_ema + prev_day_close_gap + normalized_long_term_return + volatility_volume_adjusted + volume_momentum)
    final_factor_transformed = np.log(1 + final_factor)
    
    return final_factor_transformed
