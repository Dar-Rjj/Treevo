import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-to-Low Range
    high_low_range = df['high'] - df['low']

    # Calculate Weighted Average of Price Gaps
    price_gap = df['high'] - df['low']
    volume_trend = (df['volume'].rolling(window=5).mean() * (df['volume'] / df['volume'].shift(1))).fillna(0)
    weighted_price_gaps = (price_gap * volume_trend).sum()

    # Combine High-to-Low Range with Weighted Price Gaps
    combined_value = (high_low_range * 100) + weighted_price_gaps

    # Smoothing and Initial Momentum Calculation
    smoothed_value = combined_value.rolling(window=5).mean()
    initial_momentum = smoothed_value - smoothed_value.shift(5)

    # Calculate Price Change
    price_change = df['close'] - df['open'].shift(1)

    # Final Combined Momentum
    final_combined_momentum = initial_momentum + price_change

    # Calculate Multi-Day Price Momentum
    one_day_return = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    five_day_return = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    twenty_day_return = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)

    # Average the Momentum
    average_momentum = (one_day_return + five_day_return + twenty_day_return) / 3

    # Adjust for Volume Changes
    daily_volume_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    final_volume_adjustment = average_momentum * (1 + daily_volume_change)

    # Intermediate Factor Combination
    intermediate_factor = final_combined_momentum + final_volume_adjustment

    # Calculate Weighted Price Momentum
    log_return = (df['close'] - df['close'].rolling(window=10).mean()) * (df['close'] / df['close'].shift(1)).apply(lambda x: math.log(x))

    # Calculate Rolling High-Low Range
    rolling_high = df['high'].rolling(window=30).max()
    rolling_low = df['low'].rolling(window=30).min()
    rolling_high_low_range = rolling_high - rolling_low

    # Weight by Volume Trend
    volume_trend_10 = df['volume'].rolling(window=10).mean()
    volume_weight = df['volume'] - volume_trend_10

    # Final Alpha Factor
    alpha_factor = (
        intermediate_factor +
        log_return +
        ((high_low_range * 100) + weighted_price_gaps) +
        (rolling_high_low_range * volume_weight) -
        average_momentum
    )

    return alpha_factor
