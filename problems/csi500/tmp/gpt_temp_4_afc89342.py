import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Enhanced Momentum Component
    short_term_return = df['close'].pct_change(5)
    medium_term_return = df['close'].pct_change(10)
    long_term_return = df['close'].pct_change(20)
    
    enhanced_momentum = 0.4 * short_term_return + 0.3 * medium_term_return + 0.3 * long_term_return
    
    # Calculate Intraday Movement with Volume Confirmation
    intraday_high_low_spread = df['high'] - df['low']
    intraday_close_open_change = df['close'] - df['open']
    combined_intraday_movement = (intraday_high_low_spread + intraday_close_open_change) * df['volume']
    
    # Evaluate Recent Trend
    avg_intraday_movement_5d = combined_intraday_movement.rolling(window=5).mean()
    recent_trend = combined_intraday_movement / avg_intraday_movement_5d
    
    # Calculate Liquidity Component
    avg_volume_10d = df['volume'].rolling(window=10).mean()
    price_range_5d = (df['high'] - df['low']).rolling(window=5).mean()
    
    liquidity_component = price_range_5d / avg_volume_10d
    
    # Combine Enhanced Momentum and Liquidity Components
    combined_enhanced_momentum = enhanced_momentum * price_range_5d / avg_volume_10d
    
    # Integrate Intraday Movement and Enhanced Momentum
    integrated_intraday_enhanced_momentum = recent_trend * combined_enhanced_momentum / avg_volume_10d
    
    # Calculate Volume Adjusted High-Low Range
    high_low_range = df['high'] - df['low']
    volume_adjusted_high_low_range = high_low_range * df['volume']
    
    # Calculate Volume Adjusted High-Low Range Momentum
    vol_adj_high_low_range_momentum = volume_adjusted_high_low_range.diff()
    
    # Calculate Volume Change
    volume_change = df['volume'].diff()
    
    # Combine Volume Adjusted High-Low Range Momentum and Volume Change
    adjusted_vol_adj_high_low_range_momentum = vol_adj_high_low_range_momentum * volume_change
    adjusted_vol_adj_high_low_range_momentum = adjusted_vol_adj_high_low_range_momentum.apply(lambda x: x if volume_change > 0 else -x)
    
    # Final Integration
    final_factor = integrated_intraday_enhanced_momentum * adjusted_vol_adj_high_low_range_momentum
    
    # Final Adjustment
    final_factor = final_factor ** 0.5
    stability_component = df['close'].rolling(window=10).mean()
    final_factor = final_factor - stability_component
    
    return final_factor
