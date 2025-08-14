import pandas as pd
import pandas as pd

def heuristics_v2(df, alpha=0.1, V_threshold=1.5, N=14, P=0.01):
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()

    # Apply Exponential Moving Average to Daily Returns
    df['ema'] = df['daily_return'].ewm(alpha=alpha, adjust=False).mean()

    # Apply Enhanced Volume Filter
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['accepted_ema'] = df.apply(lambda row: row['ema'] if row['volume_ratio'] > V_threshold else 0, axis=1)

    # Calculate Average True Range (ATR) over N days
    def true_range(row):
        high_low = row['high'] - row['low']
        high_close_prev = abs(row['high'] - row['close'].shift(1))
        low_close_prev = abs(row['low'] - row['close'].shift(1))
        return max(high_low, high_close_prev, low_close_prev)

    df['true_range'] = df.apply(true_range, axis=1)
    df['atr'] = df['true_range'].rolling(window=N, min_periods=1).mean()

    # Apply Enhanced Price Volatility Filter
    df['final_factor'] = df.apply(lambda row: row['accepted_ema'] if row['atr'] > P else 0, axis=1)

    return df['final_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# factor_values = heuristics_v2(df)
# print(factor_values)
