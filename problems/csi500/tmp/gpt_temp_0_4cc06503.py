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
    combined_intraday_movement = intraday_high_low_spread + intraday_close_open_change
    weighted_intraday_movement = combined_intraday_movement * df['volume']

    # Evaluate Recent Trend
    avg_weighted_intraday_movement = weighted_intraday_movement.rolling(window=5).mean()
    recent_trend = (weighted_intraday_movement - avg_weighted_intraday_movement) / avg_weighted_intraday_movement

    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    prev_high_low_range = df['high'].shift(1) - df['low'].shift(1)

    # Compute High-Low Momentum
    close_to_close_return = df['close'].pct_change()
    high_low_momentum = (high_low_range - prev_high_low_range) * close_to_close_return

    # Calculate Volume Change
    volume_change = df['volume'].pct_change()

    # Combine High-Low Momentum and Volume Change
    combined_high_low_momentum = high_low_momentum * volume_change
    combined_high_low_momentum = combined_high_low_momentum.where(volume_change > 0, -combined_high_low_momentum)

    # Volume-Weighted Adjustment
    ema_volume = df['volume'].ewm(span=5).mean()
    adjusted_combined_high_low_momentum = combined_high_low_momentum / ema_volume

    # Combine Enhanced Momentum and Intraday Movement
    integrated_intraday_enhanced_momentum = (recent_trend * enhanced_momentum) / df['volume'].rolling(window=5).mean()

    # Final Integration
    final_factor = integrated_intraday_enhanced_momentum * adjusted_combined_high_low_momentum

    # Final Adjustment
    final_factor = final_factor.apply(lambda x: x ** 0.5)
    average_close_price = df['close'].rolling(window=10).mean()
    final_factor -= average_close_price

    return final_factor
