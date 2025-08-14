import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Short-Term (5-day) and Long-Term (20-day) Moving Averages
    short_term_avg = df['close'].rolling(window=5).mean()
    long_term_avg = df['close'].rolling(window=20).mean()
    
    # Subtract Short-Term from Long-Term and Adjust for Volume
    avg_diff = (long_term_avg - short_term_avg) * df['volume']
    
    # Calculate Daily High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Compute Weighted Sum of Recent Spreads (10 days with decay factor 0.9)
    decay_factor = 0.9
    weighted_spread = 0
    for i in range(1, 11):
        weighted_spread += high_low_spread.shift(i) * (decay_factor ** i)
    high_low_spread_momentum = weighted_spread
    
    # Calculate Intraday Return
    intraday_return = (df['high'] - df['low']) / df['low']
    
    # Calculate Close-to-Open Reversal
    close_open_reversal = (df['close'] - df['open']) / df['open']
    
    # Calculate Volume Weighted Intraday Volatility
    volume_weighted_intraday_volatility = (df['high'] - df['low']) * df['volume']
    
    # Combine Intraday Factors
    combined_intraday_factors = intraday_return + close_open_reversal + volume_weighted_intraday_volatility
    
    # Adjust High-Low Spread by Volume and Lagged Close Price
    adjusted_high_low_spread = (df['high'] - df['low']) * df['volume'] / df['close'].shift(1)
    
    # Add Momentum Component
    momentum_component = df['close'] - df['close'].shift(5)
    
    # Calculate High-Low Range Ratio
    high_low_range_ratio = df['high'] / df['low']
    
    # Introduce Volatility Component using Garman-Klass volatility estimator
    daily_volatility = 0.5 * (df['high'] - df['low'])**2 - (2 * math.log(2) - 1) * (df['close'] - df['open'])**2
    final_alpha_factor = adjusted_high_low_spread * daily_volatility
    
    # Incorporate Close-Open Return Adjustment
    close_open_return = (df['close'] - df['open']) / df['open']
    final_alpha_factor *= close_open_return * (df['high'] - df['low']) * df['volume']
    
    # Final Factor Combination
    final_alpha_factor = (combined_intraday_factors + 
                          momentum_component + 
                          adjusted_high_low_spread + 
                          high_low_spread_momentum + 
                          high_low_range_ratio * df['volume'] * close_open_return)
    
    return final_alpha_factor
