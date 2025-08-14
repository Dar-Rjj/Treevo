import numpy as np
def heuristics_v2(df):
    # Calculate Short-term and Long-term Simple Moving Averages
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_diff_5_20'] = df['SMA_5'] - df['SMA_20']

    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    df['SMA_diff_10_30'] = df['SMA_10'] - df['SMA_30']

    # Calculate Short-term and Long-term Exponential Moving Averages
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['EMA_diff_10_30'] = df['EMA_10'] - df['EMA_30']

    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_diff_20_50'] = df['EMA_20'] - df['EMA_50']

    # Calculate Rate of Change (ROC)
    df['ROC_daily'] = (df['close'] / df['close'].shift(1) - 1) * 100
    df['ROC_weekly'] = (df['close'] / df['close'].shift(5) - 1) * 100

    # Calculate True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    # Calculate Average True Range (ATR)
    df['ATR_14'] = df['true_range'].rolling(window=14).mean()
    df['ATR_7'] = df['true_range'].rolling(window=7).mean()

    # Calculate Volume change rate
    df['volume_change_rate_1'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['volume_change_rate_2'] = (df['volume'] - df['volume'].shift(2)) / df['volume'].shift(2)

    # Calculate On-balance Volume (OBV)
    df['obv'] = 0
    df['obv'] = np.where(df['close'] > df['close'].shift(1), df['volume'], np.where(df['close'] < df['close'].shift(1), -df['volume'], 0)).cumsum()

    # Calculate pattern strength or frequency
    def count_candlestick_patterns(df, pattern_name):
        if pattern_name == 'Doji':
            df['doji'] = ((df['high'] - df['low']) / df['close'] < 0.005).astype(int)
            return df['doji'].rolling(window=5).sum()
