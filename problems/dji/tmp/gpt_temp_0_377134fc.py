import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Divide by Previous Close
    prev_close = df['close'].shift(1)
    momentum_high_low = high_low_spread / prev_close
    
    # Calculate Price Momentum
    recent_close = df['close']
    close_10_days_ago = df['close'].shift(10)
    price_momentum = recent_close - close_10_days_ago
    
    # Calculate Intraday Returns
    intraday_return = (df['high'] - df['low']) / df['low']
    
    # Adjust by Volume
    n_days = 10
    cumulative_volume = df['volume'].rolling(window=n_days).sum()
    adjusted_momentum = momentum_high_low / cumulative_volume
    
    # Calculate Cumulative Volume-Weighted Momentum
    volume_weighted_momentum = (adjusted_momentum * df['volume']).rolling(window=n_days).sum()
    
    # Confirm with Volume Trend
    avg_volume = df['volume'].rolling(window=10).mean()
    current_volume = df['volume']
    volume_ratio = current_volume / avg_volume
    
    if volume_ratio > 1.2:
        combined_momentum = price_momentum * intraday_return
        aggregated_momentum = volume_weighted_momentum
    else:
        combined_momentum = 0.5 * (price_momentum + intraday_return)
        aggregated_momentum = 0.5 * (volume_weighted_momentum + combined_momentum)
    
    # Calculate Intraday Price Movement
    intraday_movement = df['high'] - df['low']
    
    # Calculate Opening Price Trend Impact
    opening_trend = df['open'] - df['close'].shift(1)
    
    # Calculate Volume-Weighted Intraday Movement
    volume_weighted_intraday = intraday_movement * df['volume']
    volume_weighted_intraday_ma = volume_weighted_intraday.rolling(window=5).mean()
    
    # Calculate Amount-Weighted Opening Trend
    amount_weighted_opening = opening_trend * df['amount']
    amount_weighted_opening_ma = amount_weighted_opening.rolling(window=5).mean()
    
    # Combine Weighted Movements and Trends
    combined_factor = (volume_weighted_intraday + amount_weighted_opening) - volume_weighted_intraday_ma - amount_weighted_opening_ma
    
    return combined_factor
