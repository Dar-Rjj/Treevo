def heuristics(df):
    # Calculate Momentum Indicators
    df['5_day_SMA'] = df['close'].rolling(window=5).mean()
    df['20_day_SMA'] = df['close'].rolling(window=20).mean()
    df['50_day_SMA'] = df['close'].rolling(window=50).mean()
    df['200_day_SMA'] = df['close'].rolling(window=200).mean()
    
    df['momentum_5_20'] = df['5_day_SMA'] - df['20_day_SMA']
    df['momentum_50_200'] = df['50_day_SMA'] - df['200_day_SMA']
    
    # Calculate Volume-Based Indicators
    df['5_day_avg_volume'] = df['volume'].rolling(window=5).mean()
    df['20_day_avg_volume'] = df['volume'].rolling(window=20).mean()
    df['50_day_avg_volume'] = df['volume'].rolling(window=50).mean()
    df['200_day_avg_volume'] = df['volume'].rolling(window=200).mean()
    
    df['volume_diff_5_20'] = df['5_day_avg_volume'] - df['20_day_avg_volume']
    df['volume_diff_50_200'] = df['50_day_avg_volume'] - df['200_day_avg_volume']
    
    # Calculate Volatility Indicators
    df['5_day_std'] = df['close'].rolling(window=5).std()
    df['20_day_std'] = df['close'].rolling(window=20).std()
    df['50_day_std'] = df['close'].rolling(window=50).std()
    df['200_day_std'] = df['close'].rolling(window=200).std()
    
    df['volatility_ratio_5_20'] = df['5_day_std'] / df['20_day_std']
    df['volatility_ratio_50_200'] = df['50_day_std'] / df['200_day_std']
    
    # Calculate Relative Strength Indicators
    def rsi(series, n=14):
        delta = series.diff(1)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.rolling(n).mean()
        roll_down = down.abs().rolling(n).mean()
        RS = roll_up / roll_down
        return 100.0 - (100.0 / (1.0 + RS))
