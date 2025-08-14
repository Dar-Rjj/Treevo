import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from ta.trend import DEMAIndicator
from ta.volatility import AverageTrueRange

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Calculate Intraday High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Combine Intraday Return and High-Low Range
    combined_factor = intraday_return * high_low_range
    
    # Smooth using Double Exponential Moving Average (DEMA)
    dema = DEMAIndicator(close=combined_factor, window=14)
    smoothed_factor = dema.dema_indicator()
    
    # Apply Volume Weighting
    volume_weighted_smoothed_factor = smoothed_factor * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    previous_day_close = df['close'].shift(1)
    closing_gap = df['open'] - previous_day_close
    gap_adjusted_factor = volume_weighted_smoothed_factor + closing_gap
    
    # Integrate Long-Term Momentum
    long_term_return = df['close'] - df['close'].shift(50)
    normalized_long_term_return = long_term_return / high_low_range
    
    # Include Enhanced Dynamic Volatility Component
    rolling_std = intraday_return.rolling(window=20).std()
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    combined_volatility = (rolling_std + atr) / 2
    
    # Adjust Volatility Component with Volume
    volatility_adjusted = combined_volatility * df['volume']
    
    # Consider Liquidity
    liquidity_measure = df['volume'] / high_low_range
    final_factor = (gap_adjusted_factor + normalized_long_term_return + volatility_adjusted) * liquidity_measure
    
    # Final Factor Calculation with Non-Linear Transformation
    final_factor_transformed = np.log(1 + final_factor)
    
    return final_factor_transformed
