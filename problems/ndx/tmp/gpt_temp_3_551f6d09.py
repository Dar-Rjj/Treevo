import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Long-Term Momentum (20-day Moving Average)
    long_term_momentum = df['close'].rolling(window=20).mean()
    
    # Calculate Short-Term Momentum (5-day Moving Average)
    short_term_momentum = df['close'].rolling(window=5).mean()
    
    # Calculate Price Momentum
    price_momentum = long_term_momentum - short_term_momentum
    
    # Calculate Daily Log Return
    daily_log_return = (df['close'] / df['open']).apply(lambda x: math.log(x))
    
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Adjust Log Return by High-Low Range
    adjusted_log_return = daily_log_return / high_low_range
    
    # Filter by Volume Confirmation
    volume_growth = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    positive_volume_growth = (volume_growth > 0).astype(int)
    
    # Detect Volume Surge
    volume_surge = ((df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1) * 100) > 10
    volume_surge_indicator = volume_surge.astype(int)
    
    # Final Combined Alpha Factor
    combined_alpha_factor = price_momentum * positive_volume_growth
    
    # Incorporate Volume Surge into Final Alpha Factor
    final_combined_alpha_factor = combined_alpha_factor * volume_surge_indicator
    
    # Calculate Intraday Return
    intraday_return = (df['high'] - df['low']) / df['low']
    
    # Detect Volume Spike
    volume_spike = (df['volume'] > 2.0 * df['volume'].rolling(window=10).mean())
    volume_spike_indicator = volume_spike.astype(int)
    
    # Compute Intraday Reversal Factor
    intraday_reversal_factor = intraday_return * volume_spike_indicator
    
    # Calculate Open-to-Close Return
    open_to_close_return = (df['close'] - df['open']) / df['open']
    
    # Apply Volume Spike to Open-to-Close Return
    volume_weighted_open_to_close_return = open_to_close_return * volume_spike_indicator
    
    # Introduce Momentum Factor
    momentum_factor = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * volume_spike_indicator
    
    # Combine Alpha Factors
    combined_factors = (intraday_reversal_factor 
                        - volume_weighted_open_to_close_return 
                        + momentum_factor)
    
    # Final Alpha Factor
    final_alpha_factor = combined_factors * volume_surge_indicator
    
    return final_alpha_factor
