import pandas as pd

def heuristics_v2(df):
    def calculate_moving_averages(df, windows):
        for window in windows:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        return df

    def calculate_volatility(df, windows):
        for window in windows:
            df[f'vol_{window}'] = df['close'].pct_change().rolling(window=window).std()
        return df

    # Calculate moving averages and volatilities over different time horizons
    ma_windows = [5, 10, 20, 50]
    vol_windows = [5, 10, 20]

    df = calculate_moving_averages(df, ma_windows)
    df = calculate_volatility(df, vol_windows)

    # Construct the heuristics matrix
    heuristics_matrix = pd.DataFrame(index=df.index)
    heuristics_matrix['ma_diff_5_10'] = df['ma_5'] - df['ma_10']
    heuristics_matrix['ma_diff_10_20'] = df['ma_10'] - df['ma_20']
    heuristics_matrix['vol_ratio_5_10'] = df['vol_5'] / df['vol_10']
    heuristics_matrix['vol_ratio_10_20'] = df['vol_10'] / df['vol_20']

    return heuristics_matrix
