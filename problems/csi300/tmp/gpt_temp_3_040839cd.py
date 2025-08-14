def heuristics_v2(df):
    # Trend-based alpha factors
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_5_20_Diff'] = df['SMA_5'] - df['SMA_20']
    df['SMA_20_50_Diff'] = df['SMA_20'] - df['SMA_50']

    # Momentum-based alpha factors
    df['Return_1'] = df['close'].pct_change(1)
    df['Return_5'] = df['close'].pct_change(5)
    df['Return_20'] = df['close'].pct_change(20)
    df['Cumulative_Return_5'] = df['Return_1'].rolling(window=5).sum()
    df['Cumulative_Return_20'] = df['Return_1'].rolling(window=20).sum()

    # Volatility-based alpha factors
    df['Range_1'] = df['high'] - df['low']
    df['AverageRange_5'] = df['Range_1'].rolling(window=5).mean()
    df['AverageRange_20'] = df['Range_1'].rolling(window=20).mean()
    df['StdDev_5'] = df['Return_1'].rolling(window=5).std()
    df['StdDev_20'] = df['Return_1'].rolling(window=20).std()

    # Volume-based alpha factors
    df['VolumeChange_1'] = df['volume'].pct_change(1)
    df['VolumeAvg_5'] = df['volume'].rolling(window=5).mean()
    df['VolumeAvg_20'] = df['volume'].rolling(window=20).mean()
    df['VolumeWeightedReturn_1'] = (df['Return_1'] * df['volume']) / df['volume'].rolling(window=1).mean()
    df['VolumeWeightedReturn_5'] = (df['Return_5'] * df['volume']) / df['volume'].rolling(window=5).mean()
    df['VolumeWeightedReturn_20'] = (df['Return_20'] * df['volume']) / df['volume'].rolling(window=20).mean()

    # Pattern-based alpha factors
    df['Doji'] = (abs(df['close'] - df['open']) < 0.5 * (df['high'] - df['low'])).astype(int)
    df['Hammer'] = (((df['close'] - df['low']) / (df['high'] - df['low'])) > 0.6) & ((df['close'] - df['open']) > 0)
    df['ShootingStar'] = (((df['high'] - df['close']) / (df['high'] - df['low'])) > 0.6) & ((df['open'] - df['close']) > 0)

    df['PriceBreakout_5'] = (df['close'] > df['high'].rolling(window=5).max()).astype(int)
    df['PriceBreakdown_5'] = (df['close'] < df['low'].rolling(window=5).min()).astype(int)

    # Strength-based alpha factors
    def rsi(data, window):
        delta = data.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.rolling(window).mean()
        roll_down = down.abs().rolling(window).mean()
        rs = roll_up / roll_down
        return 100.0 - (100.0 / (1.0 + rs))
