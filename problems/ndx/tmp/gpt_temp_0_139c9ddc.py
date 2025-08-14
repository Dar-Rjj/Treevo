import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Short-Term Momentum
    short_term_momentum = df['close'].pct_change(periods=10)
    
    # Calculate Medium-Term Momentum
    medium_term_momentum = df['close'].pct_change(periods=30)
    
    # Calculate Long-Term Momentum
    long_term_momentum = df['close'].pct_change(periods=60)
    
    # Adjust by Volatility and Trading Volume
    # Calculate Volatility
    volatility = df['close'].rolling(window=20).std()
    
    # Calculate Average Trading Volume
    avg_volume = df['volume'].rolling(window=20).mean()
    
    # Generate Complexity Score
    combined_momentum = (short_term_momentum / volatility + 
                         medium_term_momentum / volatility + 
                         long_term_momentum / volatility) * avg_volume
    
    # Identify Volume Spikes
    volume_20_day_ma = df['volume'].rolling(window=20).mean()
    volume_spike = df['volume'] > 1.5 * volume_20_day_ma
    
    # Identify Price Spikes
    close_20_day_ma = df['close'].rolling(window=20).mean()
    price_spike = df['close'] > 1.5 * close_20_day_ma
    
    # Combine Adjusted Momentum, Volume Spike, and Price Spike
    multiplier = 1
    multiplier[volume_spike & price_spike] = 3
    multiplier[volume_spike & ~price_spike] = 2
    multiplier[~volume_spike & price_spike] = 1.5
    
    final_factor_value = combined_momentum * multiplier
    
    return final_factor_value
