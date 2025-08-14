import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Enhanced High-to-Low Range
    high_low_range = df['high'] - df['low']

    # Calculate Price Change
    price_change = df['close'].shift(-1) - df['open']

    # Combine Enhanced High-to-Low Range with Price Change
    combined_value = (high_low_range * 100) + price_change

    # Smoothing and Initial Momentum Calculation
    smoothed_momentum = combined_value.ewm(span=7).mean()
    initial_momentum = smoothed_momentum - smoothed_momentum.shift(7)

    # Calculate Daily Log Return
    daily_log_return = np.log(df['close'] / df['close'].shift(1))

    # Calculate Multi-Day Price Momentum
    three_day_return = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    seven_day_return = (df['close'] - df['close'].shift(7)) / df['close'].shift(7)
    twenty_one_day_return = (df['close'] - df['close'].shift(21)) / df['close'].shift(21)

    # Weighted Average the Momentum
    weighted_momentum = (3 * three_day_return + 2 * seven_day_return + twenty_one_day_return) / 6

    # Adjust for Volume Changes
    volume_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    final_volume_adjustment = weighted_momentum * (1 + 0.5 * volume_change)

    # Intermediate Factor Combination
    intermediate_factor = initial_momentum + final_volume_adjustment

    # Calculate Volume-Weighted Price Momentum
    five_day_wma = df['close'].rolling(window=5).mean().shift(1)
    volume_weighted_momentum = (df['close'] - five_day_wma) * daily_log_return

    # Introduce High-Frequency Volatility
    true_range = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2].shift(1)), abs(x[1] - x[2].shift(1))), axis=1)
    sma_true_range = true_range.rolling(window=14).mean()

    # Adjust for High-Frequency Volatility
    high_freq_vol_adj_momentum = volume_weighted_momentum / sma_true_range

    # Final Factor Construction
    final_factor = intermediate_factor + high_freq_vol_adj_momentum

    return final_factor
