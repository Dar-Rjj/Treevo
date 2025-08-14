import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']

    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']

    # Determine Volatility using High, Low, and Close prices
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x) - min(x), axis=1)
    df['volatility'] = df['true_range'].rolling(window=20).std()

    # Adjust Window Size based on Volatility
    volatility_threshold = df['volatility'].median()
    window_size = df['volatility'].apply(lambda x: 10 if x > volatility_threshold else 30)

    # Create a rolling mean and standard deviation with adaptive window size
    def rolling_apply(df, window_col, func):
        result = pd.Series(index=df.index)
        for i in range(2 * window_col.max(), len(df)):
            window = window_col.iloc[i]
            result.iloc[i] = func(df.iloc[i - window:i])
        return result

    df['rolling_mean'] = rolling_apply(df, window_size, lambda x: x['volume_weighted_return'].mean())
    df['rolling_std'] = rolling_apply(df, window_size, lambda x: x['volume_weighted_return'].std())

    # Final factor is the rolling mean adjusted by the rolling standard deviation
    df['factor'] = df['rolling_mean'] / df['rolling_std']

    return df['factor']
