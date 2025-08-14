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
    
    # Smooth using Exponential Moving Average (EMA) with period 14
    smoothed_factor = combined_factor.ewm(span=14).mean()
    
    # Apply Volume Weighting
    volume_weighted_factor = smoothed_factor * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    prev_day_close_gap = df['open'].diff().fillna(0)
    adjusted_volume_weighted_factor = volume_weighted_factor + prev_day_close_gap
    
    # Integrate Long-Term Momentum
    long_term_return = df['close'] - df['close'].shift(50)
    normalized_long_term_momentum = long_term_return / high_low_range
    
    # Include Enhanced Dynamic Volatility Component
    rolling_std = intraday_return.rolling(window=20).std()
    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    combined_volatility = (rolling_std + atr) / 2
    
    # Adjust Volatility Component with Volume
    volatility_component = combined_volatility * df['volume']
    
    # Final Factor Calculation
    final_factor = (
        adjusted_volume_weighted_factor + 
        normalized_long_term_momentum +
        volatility_component
    )
    
    # Apply Non-Linear Transformation
    final_factor = np.log(1 + final_factor)
    
    return final_factor
