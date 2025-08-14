import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = df['close'] - df['open']

    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']

    # Adjust High-Low Spread by Open Price
    adjusted_high_low_spread = high_low_spread - df['open']

    # Combine Intraday Return and Adjusted High-Low Spread
    combined_factor = intraday_return * adjusted_high_low_spread

    # Volume Weighting
    volume_weighted_combination = combined_factor * df['volume']

    # Detect Volume Spike
    avg_5_day_volume = df['volume'].rolling(window=5).mean()
    volume_spike = (df['volume'] > avg_5_day_volume)

    # Apply Inverse Volume Weighting
    inverse_volume_weighting = volume_spike.apply(lambda x: 1 / (df['volume'] / avg_5_day_volume) if x else 1)

    # Adjust Volume-Weighted Combination by Inverse Volume Weighting
    adjusted_volume_weighted_combination = volume_weighted_combination * inverse_volume_weighting

    # Calculate Daily Momentum
    daily_momentum = df['close'] - df['close'].shift(1)

    # Combine Daily Momentum and Adjusted Volume-Weighted Combination
    momentum_adjusted_factor = daily_momentum * adjusted_volume_weighted_combination

    # Filter Positive Values
    positive_values = momentum_adjusted_factor.apply(lambda x: x if x > 0 else 0)

    # Final Alpha Factor
    final_alpha_factor = positive_values.rolling(window=21).sum()

    # Calculate Volume-Weighted Close-to-Open Return
    close_to_open_return = (df['close'] - df['open']) * df['volume']
    total_5_day_volume = df['volume'].rolling(window=5).sum()
    volume_weighted_close_to_open_return = close_to_open_return / total_5_day_volume

    # Combine and Weight
    combined_and_weighted_factor = (high_low_spread + volume_weighted_close_to_open_return) * (1 / df['volume'].rolling(window=10).mean())

    # Integrate All Factors
    integrated_alpha_factor = final_alpha_factor * combined_and_weighted_factor

    # Calculate Moving Average Convergence Divergence (MACD)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26

    # Incorporate MACD into the Alpha Factor
    final_integrated_alpha_factor = integrated_alpha_factor * macd

    return final_integrated_alpha_factor
