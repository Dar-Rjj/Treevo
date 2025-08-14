import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Multi-Term Momentum
    short_term_return = df['close'].pct_change(5)
    medium_term_return = df['close'].pct_change(10)
    long_term_return = df['close'].pct_change(20)
    combined_returns = 0.4 * short_term_return + 0.3 * medium_term_return + 0.3 * long_term_return

    # Calculate Intraday and High-Low Movements
    intraday_high_low_spread = df['high'] - df['low']
    intraday_close_open_change = df['close'] - df['open']
    sum_intraday_movements = intraday_high_low_spread + intraday_close_open_change
    weighted_movement = sum_intraday_movements * df['volume']
    recent_trend = weighted_movement.rolling(window=7).mean()
    trend_difference = weighted_movement - recent_trend

    # High-Low Range Momentum
    current_high_low_range = df['high'] - df['low']
    previous_high_low_range = df['high'].shift(1) - df['low'].shift(1)
    high_low_momentum = current_high_low_range - previous_high_low_range

    # Evaluate Momentum and Volume Interaction
    close_to_close_return = df['close'].pct_change()
    adjusted_high_low_momentum = high_low_momentum * close_to_close_return
    volume_change = df['volume'].diff()
    volume_interaction_factor = adjusted_high_low_momentum * (1 if volume_change > 0 else -1)

    # Calculate 20-day Weighted Moving Average of Returns
    daily_returns = df['close'].pct_change()
    wma_20 = daily_returns.rolling(window=20).mean()

    # Combine Momentum, Intraday, and Volume Components
    combined_factors = combined_returns + weighted_movement + high_low_momentum
    combined_factors *= volume_interaction_factor

    # Enhance Momentum Component
    enhanced_momentum = 0.4 * short_term_return + 0.3 * medium_term_return + 0.3 * long_term_return

    # Calculate Liquidity Component
    average_volume = df['volume'].rolling(window=30).mean()
    amplitude_of_price_movement = (df['high'] - df['low']).rolling(window=20).mean()

    # Combine Components
    combined_components = enhanced_momentum * trend_difference * amplitude_of_price_movement / average_volume

    # Detect and Adjust for Volume Spikes
    volume_spike = df['volume'] > df['volume'].rolling(window=20).mean()
    adjusted_wma_20 = wma_20.where(~volume_spike, wma_20 * 0.7)

    # Final Adjustment
    final_factor = combined_components + adjusted_wma_20
    final_factor = final_factor.apply(lambda x: x ** 0.5) + df['close'].rolling(window=30).mean()

    return final_factor
