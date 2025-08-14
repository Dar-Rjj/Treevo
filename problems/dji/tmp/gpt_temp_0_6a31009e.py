import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # High-Low Range Ratio
    df['daily_high_low_range'] = df['high'] - df['low']
    df['recent_avg_high_low_range'] = df['daily_high_low_range'].rolling(window=20).mean()
    df['high_low_range_ratio'] = df['daily_high_low_range'] / df['recent_avg_high_low_range']

    # Breakout Strength
    df['breakout_strength'] = df['daily_high_low_range'] / df['recent_avg_high_low_range']

    # Volume Weighted Momentum
    df['price_change'] = df['close'].pct_change()
    df['volume_weighted_momentum'] = df['price_change'] * df['volume']

    # Combine Breakout and Momentum
    df['combined_breakout_momentum'] = (df['breakout_strength'] + df['volume_weighted_momentum']) / 2

    # Price Momentum
    df['7_day_return'] = df['close'].pct_change(7)
    df['21_day_return'] = df['close'].pct_change(21)

    # Volume Activity
    df['7_day_avg_volume'] = df['volume'].rolling(window=7).mean()
    df['21_day_avg_volume'] = df['volume'].rolling(window=21).mean()

    # Momentum Indicators
    df['SMA'] = df['close'].rolling(window=20).mean()
    df['EMA'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ROC'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100

    # Volume Impact
    df['volume_ratio'] = df['volume'] / df['21_day_avg_volume']
    weighted_indicators = (df['SMA'] + df['EMA'] + df['ROC']) / 3 * df['volume_ratio']

    # Trend Indicator
    df['DM_plus'] = df['high'].diff().clip(lower=0)
    df['DM_minus'] = -df['low'].diff().clip(upper=0)
    df['TR'] = df[['high' - 'low', 'high' - df['close'].shift(), df['close'].shift() - df['low']]].max(axis=1)
    df['plus_DI'] = df['DM_plus'].rolling(window=14).sum() / df['TR'].rolling(window=14).sum() * 100
    df['minus_DI'] = df['DM_minus'].rolling(window=14).sum() / df['TR'].rolling(window=14).sum() * 100
    df['DX'] = (df['plus_DI'] - df['minus_DI']).abs() / (df['plus_DI'] + df['minus_DI']) * 100
    df['ADX'] = df['DX'].ewm(span=14, adjust=False).mean()

    # Market Sentiment
    df['5_day_open_price_diff'] = df['open'] - df['open'].shift(5)
    df['20_day_open_price_diff'] = df['open'] - df['open'].shift(20)

    # Final Alpha Factor
    alpha_factor = (
        (df['combined_breakout_momentum'] + df['7_day_return'] + df['21_day_return'] + df['7_day_avg_volume'] + df['21_day_avg_volume']) / 5 +
        weighted_indicators + df['ADX'] + df['5_day_open_price_diff'] + df['20_day_open_price_diff']
    )

    return alpha_factor
