import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Returns
    df['daily_return'] = df['close'].pct_change()

    # Compute Moving Average of Daily Returns
    n = 20  # Lookback period
    df['ma_daily_return'] = df['daily_return'].rolling(window=n).mean()

    # Calculate Volume Weight
    df['average_volume'] = df['volume'].rolling(window=n).mean()
    df['volume_weight'] = df['volume'] / df['average_volume']

    # Combine Moving Average and Volume Weight
    df['volume_weighted_ma'] = df['ma_daily_return'] * df['volume_weight']

    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']

    # Calculate Price Breakout Ratio
    df['price_breakout_ratio'] = (df['high'] - df['open']) / df['high_low_range']

    # Calculate Volume-Weighted Breakout Indicator
    df['breakout_indicator'] = (df['close'] - df['open']) * df['volume'] * df['price_breakout_ratio']

    # Aggregate Indicators
    df['aggregate_breakout'] = df['breakout_indicator'].rolling(window=n).sum()

    # Generate Alpha Factor
    df['alpha_factor'] = df['volume_weighted_ma'] + df['aggregate_breakout']

    # Apply Moving Average to Alpha Factor
    df['final_alpha_factor'] = df['alpha_factor'].rolling(window=n).mean()

    # Adjust for Volatility
    df['volatility'] = df[['high', 'low', 'close']].pct_change().rolling(window=n).std() * np.sqrt(252)
    df['normalized_alpha_factor'] = df['final_alpha_factor'] / df['volatility']

    return df['normalized_alpha_factor'].dropna()
