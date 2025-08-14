import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Directionally Adjusted High-Low Spread
    high_low_spread = df['high'] - df['low']
    directional_bias = (df['close'] > df['open']).astype(int) * 2 - 1
    adjusted_high_low_spread = high_low_spread * directional_bias
    
    # Compute Volume-Adjusted Momentum with Price Range
    momentum = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    average_price_range = (df['high'] - df['low']).rolling(window=10).mean()
    volume_adjusted_momentum = momentum * df['volume'] / average_price_range
    
    # Calculate Volume-Weighted Return with Spike Adjustment
    high_low_return = (df['high'] - df['low']) / df['low']
    volume_weighted_return = high_low_return * df['volume']
    
    # Identify Volume Spike Days
    moving_average_volume = df['volume'].rolling(window=14).mean()
    volume_spike = (df['volume'] > 1.5 * moving_average_volume).astype(int)
    volume_weighted_return = volume_weighted_return * (1 + volume_spike)
    
    # Calculate Intraday Momentum
    intraday_momentum = df['high'] - df['low']
    
    # Calculate Enhanced Intraday Volatility
    enhanced_intraday_volatility = (df['high'] - df['low']) * 1.5
    
    # Integrate Adjusted High-Low Spread, Volume-Weighted Return, and Volume-Adjusted Momentum
    integrated_factor = (adjusted_high_low_spread * volume_weighted_return) + volume_adjusted_momentum
    
    # Apply Final Volume Momentum Modifier
    average_20_day_volume = df['volume'].rolling(window=20).mean()
    close_to_average_volume_ratio = df['close'] / average_20_day_volume
    integrated_factor *= close_to_average_volume_ratio
    
    # Adjust Factor by Open-Close Trend
    open_close_trend = df['close'] - df['open']
    integrated_factor = integrated_factor * (1.2 if open_close_trend > 0 else 0.8)
    
    # Apply Directional Bias to Integrated Factor
    integrated_factor = integrated_factor * (1.5 if df['close'] > df['open'] else 0.5)
    
    # Calculate Moving Averages
    short_term_ma = df['close'].rolling(window=7).mean()
    long_term_ma = df['close'].rolling(window=30).mean()
    
    # Compute Crossover Signal
    crossover_signal = short_term_ma - long_term_ma
    
    # Generate Alpha Factor
    alpha_factor = crossover_signal.apply(lambda x: 1 if x > 0 else -1)
    
    return alpha_factor
