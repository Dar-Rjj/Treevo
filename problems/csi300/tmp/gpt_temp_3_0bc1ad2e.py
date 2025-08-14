import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['close'].shift(-1) - df['open']

    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']

    # Determine Volatility using High, Low, and Close prices
    df['high_low_close_mean'] = (df['high'] + df['low'] + df['close']) / 3
    volatility = df['high_low_close_mean'].rolling(window=20).std()

    # Adjust Window Size Based on Volatility
    def adaptive_window_size(volatility):
        if volatility > volatility.quantile(0.75):
            return 5  # Decrease window size for high volatility
        elif volatility < volatility.quantile(0.25):
            return 20  # Increase window size for low volatility
        else:
            return 10  # Default window size

    df['adaptive_window'] = volatility.apply(adaptive_window_size)

    # Calculate Rolling Mean and Standard Deviation with Adaptive Window
    df['rolling_mean'] = df.groupby('adaptive_window')['volume_weighted_return'].apply(lambda x: x.rolling(window=x.name).mean())
    df['rolling_std'] = df.groupby('adaptive_window')['volume_weighted_return'].apply(lambda x: x.rolling(window=x.name).std())

    # Volatility Adjustment
    df['volatility_adjusted_factor'] = df['rolling_mean'] / df['rolling_std']

    # Return the alpha factor
    return df['volatility_adjusted_factor']
