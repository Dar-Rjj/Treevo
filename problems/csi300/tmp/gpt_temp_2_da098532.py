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
    volume_weighted_smoothed_factor = smoothed_factor * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    previous_closing_gap = df['open'].shift(-1) - df['close']
    volume_weighted_smoothed_factor += previous_closing_gap
    
    # Integrate Long-Term Momentum
    long_term_return = df['close'] - df['close'].shift(50)
    normalized_long_term_return = long_term_return / high_low_range
    
    # Include Enhanced Dynamic Volatility Component
    rolling_std = intraday_return.rolling(window=20).std()
    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    combined_volatility = (rolling_std + atr) / 2
    volatility_adjusted = combined_volatility * df['volume']
    
    # Final Factor Calculation
    final_factor = (volume_weighted_smoothed_factor + 
                    previous_closing_gap + 
                    normalized_long_term_return + 
                    volatility_adjusted)
    
    # Apply Non-Linear Transformation
    final_factor = np.log1p(final_factor)
    
    return final_factor
