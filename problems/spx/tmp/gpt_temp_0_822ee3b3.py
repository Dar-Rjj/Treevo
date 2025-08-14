import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=10):
    # Calculate Daily Price Jerk
    df['close_t'] = df['close']
    df['close_t-1'] = df['close'].shift(1)
    df['close_t-n'] = df['close'].shift(n)
    df['third_diff'] = df['close_t'] - 3 * df['close_t-1'] + 3 * df['close_t-n'] - df['close_t-1'].shift(n-1)
    df['smoothed_jerk'] = df['third_diff'].rolling(window=n).sum()

    # Incorporate Volume Adjusted Inertia
    df['positive_volume'] = df.apply(lambda row: row['volume'] * (row['close'] - df.loc[row.name - pd.Timedelta(days=1), 'close']) if row['close'] > df.loc[row.name - pd.Timedelta(days=1), 'close'] else 0, axis=1)
    df['negative_volume'] = df.apply(lambda row: -row['volume'] * (df.loc[row.name - pd.Timedelta(days=1), 'close'] - row['close']) if row['close'] <= df.loc[row.name - pd.Timedelta(days=1), 'close'] else 0, axis=1)
    df['sum_positive_volume'] = df['positive_volume'].rolling(window=n).sum()
    df['sum_negative_volume'] = df['negative_volume'].rolling(window=n).sum().abs()
    df['volume_adjusted_inertia'] = df['smoothed_jerk'] * df['sum_positive_volume'] / (df['sum_negative_volume'] + 1e-6)  # Avoid division by zero

    # Integrate Enhanced Price-Volume and Spread Dynamics
    df['high_low_spread'] = df['high'] - df['low']
    df['open_close_spread'] = df['open'] - df['close']
    df['pearson_corr'] = df['close'].rolling(window=n).corr(df['volume'])
    df['spearman_corr'] = df['high_low_spread'].rolling(window=n).apply(lambda x: x.corr(df.loc[x.index, 'volume'], method='spearman'))
    
    # Calculate Moving Averages
    df['avg_high_low_spread'] = df['high_low_spread'].rolling(window=n).mean()
    df['avg_open_close_spread'] = df['open_close_spread'].rolling(window=n).mean()
    
    df['relative_high_low_spread'] = df['high_low_spread'] / (df['avg_high_low_spread'] + 1e-6)  # Avoid division by zero
    df['relative_open_close_spread'] = df['open_close_spread'] / (df['avg_open_close_spread'] + 1e-6)  # Avoid division by zero
    
    # Final Alpha Factor
    df['alpha_factor'] = df['volume_adjusted_inertia'] * df['pearson_corr'] * df['spearman_corr'] * df['relative_high_low_spread'] * df['relative_open_close_spread']
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
