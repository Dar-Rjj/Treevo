import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Enhanced High-to-Low Price Ratio
    df['high_low_ratio'] = df['high'] / df['low']

    # Evaluate Volume Trend
    window = 10
    df['volume_ma'] = df['volume'].rolling(window=window).mean()
    df['volume_trend'] = np.where(df['volume'] > df['volume_ma'], 1, -1)

    # Quantify High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    df['true_range'] = df[['high', 'close']].shift(1).apply(lambda x: max(x['high'] - df['low'], abs(x['high'] - x['close']), abs(df['low'] - x['close'])), axis=1)
    df['avg_true_range'] = df['true_range'].rolling(window=window).mean()

    # Combine Volume Trend and Price Volatility
    df['std_true_range'] = df['true_range'].rolling(window=window).std()

    # Integrate Components for Alpha Factor
    df['adjusted_high_low_ratio'] = df['high_low_ratio'] * df['volume_trend']
    df['adjusted_high_low_range'] = df['high_low_range'] * df['volume_trend']
    df['inverted_volume_trend'] = -df['volume_trend']
    df['alpha_factor'] = df['adjusted_high_low_ratio'] + df['inverted_volume_trend'] + df['std_true_range']

    return df['alpha_factor']
