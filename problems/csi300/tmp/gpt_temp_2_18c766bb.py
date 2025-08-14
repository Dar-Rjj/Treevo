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
    
    # Adaptive EMA period based on recent volatility
    rolling_std = df['close'].rolling(window=20).std()
    avg_true_range = (df['high'] - df['low']).rolling(window=14).mean()
    dynamic_volatility = (rolling_std + avg_true_range) / 2
    adaptive_ema_period = 20 / (1 + dynamic_volatility)
    smoothed_factor = combined_factor.ewm(span=adaptive_ema_period, adjust=False).mean()
    
    # Apply Volume Weighting
    volume_weighted_factor = smoothed_factor * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    previous_day_close_gap = df['open'] - df['close'].shift(1)
    gap_adjusted_factor = volume_weighted_factor + previous_day_close_gap
    
    # Integrate Long-Term Momentum
    long_term_return = df['close'] - df['close'].shift(50)
    normalized_long_term_return = long_term_return / high_low_range
    
    # Include Sector-Specific Momentum
    # Assuming 'sector_index' is a column in the DataFrame
    sector_return = df['sector_index'] - df['sector_index'].shift(50)
    sector_adjusted_return = normalized_long_term_return / sector_return
    
    # Include Enhanced Dynamic Volatility Component
    combined_volatility = (rolling_std + avg_true_range) / 2
    volume_adjusted_volatility = combined_volatility * df['volume']
    
    # Include Liquidity Measure
    turnover_ratio = df['volume'] / df['amount']
    
    # Final Factor Calculation
    final_factor = (
        gap_adjusted_factor +
        normalized_long_term_return +
        sector_adjusted_return +
        volume_adjusted_volatility
    ) * turnover_ratio
    
    # Apply Non-Linear Transformation
    final_factor = np.log(1 + final_factor)
    
    return final_factor
