import pandas as pd
import pandas as pd

def heuristics_v2(df, window_size):
    # Calculate daily return
    df['daily_return'] = df['close'].pct_change()

    # Apply cumulative sum over the window
    df['cumulative_return'] = df['daily_return'].rolling(window=window_size).sum()

    # Integrate volume as a weight factor
    df['volume_moving_avg'] = df['volume'].rolling(window=window_size).mean()
    df['volume_ratio'] = df['volume'] / df['volume_moving_avg']

    # Combine weighted returns
    df['weighted_cumulative_return'] = df['cumulative_return'] * df['volume_ratio']

    # Return the alpha factor as a pandas Series
    return df['weighted_cumulative_return'].dropna()
