import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Directionally Adjusted High-Low Spread
    high_low_spread = df['high'] - df['low']
    directional_bias = (df['close'] > df['open']).astype(int) * 2 - 1
    adjusted_high_low_spread = high_low_spread * directional_bias
    
    # Compute Volume-Adjusted Momentum with Price Range
    momentum = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    average_price_range = (df['high'] - df['low']).rolling(window=10).mean()
    volume_adjusted_momentum = (momentum * df['volume']) / average_price_range
    
    # Calculate Volume-Adjusted Daily Returns
    daily_returns = df['close'] - df['close'].shift(1)
    volume_14_day_ma = df['volume'].rolling(window=14).mean()
    volume_spike = (df['volume'] > 1.5 * volume_14_day_ma).astype(int)
    volume_adjusted_returns = daily_returns * (1 + volume_spike)
    
    # Integrate Adjusted High-Low Spread and Volume-Adjusted Returns
    integrated_factor = adjusted_high_low_spread * volume_adjusted_returns
    
    # Apply Volume Momentum Modifier
    aggregated_volume = df['volume'].rolling(window=14).sum()
    volume_momentum_modifier = df['close'] / aggregated_volume
    integrated_factor *= volume_momentum_modifier
    
    # Calculate Intraday Momentum
    intraday_range = df['high'] - df['low']
    
    # Calculate Volume Spike
    previous_volume = df['volume'].shift(1)
    intraday_volume_spike = (df['volume'] > 1.5 * previous_volume).astype(int) * 2 - 1
    intraday_momentum = intraday_range * (1 + intraday_volume_spike)
    
    # Aggregate Factors
    integrated_factor += intraday_momentum
    
    # Apply Directional Bias to Integrated Factor
    integrated_factor = integrated_factor * (1.5 if df['close'] > df['open'] else 0.5)
    
    # Incorporate Open-Close Trend
    open_close_trend = df['close'] - df['open']
    integrated_factor = integrated_factor * (1.2 if open_close_trend > 0 else 0.8)
    
    # Calculate Moving Averages
    short_term_ma = df['close'].rolling(window=5).mean()
    long_term_ma = df['close'].rolling(window=20).mean()
    
    # Compute Crossover Signal
    crossover_signal = short_term_ma - long_term_ma
    
    # Generate Alpha Factor
    alpha_factor = (crossover_signal > 0).astype(int) * 2 - 1
    alpha_factor = alpha_factor * integrated_factor
    
    return alpha_factor
