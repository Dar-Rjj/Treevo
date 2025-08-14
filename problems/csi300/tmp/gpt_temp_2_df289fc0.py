import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = df['close'] - df['open']
    
    # Calculate Intraday High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Combine Intraday Return and High-Low Range
    df['combined_factor'] = df['intraday_return'] * df['high_low_range']
    
    # Smooth using EMA (14 days)
    df['smoothed_factor'] = df['combined_factor'].ewm(span=14).mean()
    
    # Apply Volume Weighting
    df['volume_weighted_factor'] = df['smoothed_factor'] * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    df['prev_close_gap'] = df['open'] - df['close'].shift(1)
    df['gap_adjusted_factor'] = df['volume_weighted_factor'] + df['prev_close_gap']
    
    # Integrate Long-Term Momentum
    df['long_term_return'] = df['close'] - df['close'].shift(50)
    df['normalized_long_term_return'] = df['long_term_return'] / df['high_low_range']
    
    # Include Enhanced Dynamic Volatility Component
    df['intraday_returns_rolling_std'] = df['intraday_return'].rolling(window=20).std()
    df['atr'] = df[['high', 'low', df['close'].shift(1)].max(axis=1) - [df['low'], df['close'].shift(1), df['high']].min(axis=1)].ewm(span=14).mean()
    df['combined_volatility'] = (df['intraday_returns_rolling_std'] + df['atr']) / 2
    
    # Adjust Volatility Component with Volume
    df['volume_adjusted_volatility'] = df['combined_volatility'] * df['volume']
    
    # Final Factor Calculation
    df['final_factor'] = df['gap_adjusted_factor'] + df['normalized_long_term_return'] + df['volume_adjusted_volatility']
    
    # Apply Non-Linear Transformation
    df['non_linear_factor'] = np.log(1 + df['final_factor'])
    
    return df['non_linear_factor']
