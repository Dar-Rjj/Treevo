import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Typical Price
    typical_price = (df['high'] + df['low'] + df['close']) / 3

    # VWAP and Its Differences
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    vwap_diff_from_open = vwap - df['open']
    vwap_diff_from_close = vwap - df['close']

    # Raw Returns
    daily_return = (df['close'] / df['close'].shift(1)) - 1
    weekly_return = (df['close'] / df['close'].shift(5)) - 1
    monthly_return = (df['close'] / df['close'].shift(21)) - 1

    # Momentum Indicators
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    ema_momentum = ema_12 - ema_26
    short_term_price_change = df['close'] - df['close'].shift(15)
    long_term_price_change = df['close'] - df['close'].shift(70)

    # Aggregate Momentum Score
    aggregate_momentum = daily_return + weekly_return + monthly_return + ema_momentum

    # Volume Indicators
    obv = (df['close'].diff() > 0).astype(int) * df['volume']
    obv_adjusted = obv.cumsum() * df['close']
    volume_change = df['volume'] - df['volume'].rolling(window=30).mean()
    vpt = ((df['close'].pct_change()) * df['volume']).cumsum()

    # Combined Volume Factor
    combined_volume_factor = obv_adjusted + volume_change + vpt

    # Relative Strength
    upward_returns = (daily_return > 0) * daily_return
    downward_returns = (daily_return < 0) * daily_return.abs()
    relative_strength = upward_returns.rolling(window=20).sum() / downward_returns.rolling(window=20).sum()
    smoothed_relative_strength = relative_strength * df['volume'].ewm(span=20, adjust=False).mean()

    # Volatility Indicators
    short_term_volatility = daily_return.rolling(window=10).std()
    long_term_volatility = daily_return.rolling(window=60).std()

    # Mean Reversion Scores
    rolling_5d_mean = df['close'].rolling(window=5).mean()
    rolling_5d_std = df['close'].rolling(window=5).std()
    mean_reversion_5d = (daily_return < (rolling_5d_mean - rolling_5d_std)).astype(float)

    rolling_20d_mean = df['close'].rolling(window=20).mean()
    rolling_20d_std = df['close'].rolling(window=20).std()
    mean_reversion_20d = (daily_return > (rolling_20d_mean + rolling_20d_std)).astype(float)

    # Price Trend Adjustment
    sma_21 = df['close'].rolling(window=21).mean()
    price_trend_adjustment = df['close'] / sma_21

    # VWAP Differences Adjustment
    vwap_diff_from_open_adj = vwap_diff_from_open * smoothed_relative_strength
    vwap_diff_from_close_adj = vwap_diff_from_close * smoothed_relative_strength

    # Final Alpha Factor
    final_alpha_factor = (
        -long_term_momentum
        + short_term_momentum
        + vwap_diff_from_open_adj
        + vwap_diff_from_close_adj
        + combined_volume_factor
    )

    # Enhanced Alpha Factor
    high_10d = df['high'].rolling(window=10).max()
    low_10d = df['low'].rolling(window=10).min()
    high_low_ratio = high_10d / low_10d

    enhanced_alpha_factor = (final_alpha_factor * smoothed_relative_strength) + high_low_ratio

    return enhanced_alpha_factor
