import pandas as pd
import pandas as pd

def heuristics(df):
    # Calculate Short, Medium, and Long-Term Returns
    short_term_return = df['close'].pct_change(5)
    medium_term_return = df['close'].pct_change(10)
    long_term_return = df['close'].pct_change(20)
    
    # Enhance Momentum Component
    enhanced_momentum = 0.4 * short_term_return + 0.3 * medium_term_return + 0.3 * long_term_return
    
    # Calculate Intraday Movement
    intraday_high_low_spread = df['high'] - df['low']
    intraday_close_open_change = df['close'] - df['open']
    weighted_intraday_movement = (intraday_high_low_spread + intraday_close_open_change) * df['volume']
    
    # Evaluate Recent Trend
    avg_weighted_intraday_movement_5d = weighted_intraday_movement.rolling(window=5).mean()
    recent_trend = weighted_intraday_movement - avg_weighted_intraday_movement_5d
    
    # Calculate Liquidity Component
    avg_volume_10d = df['volume'].rolling(window=10).mean()
    price_amplitude_5d = df['close'].rolling(window=5).std()
    
    # Integrate Momentum and Liquidity Components
    integrated_momentum_liquidity = (enhanced_momentum * price_amplitude_5d) / avg_volume_10d
    
    # Integrate Intraday Movement and Enhanced Momentum
    integrated_intraday_momentum = (recent_trend * enhanced_momentum) / avg_volume_10d
    
    # Calculate Volume Adjusted High-Low Range
    volume_adjusted_high_low_range = (df['high'] - df['low']) * df['volume']
    
    # Calculate Volume Adjusted High-Low Range Momentum
    volume_adjusted_high_low_range_momentum = volume_adjusted_high_low_range - volume_adjusted_high_low_range.shift(1)
    
    # Calculate Volume Change
    volume_change = df['volume'] - df['volume'].shift(1)
    
    # Combine Volume Adjusted High-Low Range Momentum and Volume Change
    combined_momentum_volume = volume_adjusted_high_low_range_momentum * volume_change
    combined_momentum_volume = combined_momentum_volume.where(volume_change > 0, -combined_momentum_volume)
    
    # Calculate Daily Returns
    daily_returns = df['close'].pct_change()
    
    # Calculate 20-day Weighted Moving Average of Returns
    wma_20d = daily_returns.rolling(window=20).mean()
    
    # Adjust for Volume Spikes
    volume_spike_days = df['volume'] > df['volume'].rolling(window=20).mean()
    adjusted_wma_20d = wma_20d.where(~volume_spike_days, wma_20d * 0.6)
    
    # Add Adjusted 20-day Weighted Moving Average to Combined High-Low Range Momentum
    final_factor = integrated_intraday_momentum + combined_momentum_volume + adjusted_wma_20d
    
    # Apply Non-Linear Transformation
    final_factor = final_factor.apply(lambda x: x ** 0.5)
    
    # Add Stability Component
    avg_close_price_10d = df['close'].rolling(window=10).mean()
    final_factor = final_factor - avg_close_price_10d
    
    return final_factor
