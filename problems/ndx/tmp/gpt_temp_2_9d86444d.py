def heuristics_v2(df):
    # Short-Term Momentum
    df['10_day_momentum'] = df['close'].pct_change(10)
    df['20_day_momentum'] = df['close'].pct_change(20)
    df['50_day_momentum'] = df['close'].pct_change(50)
    
    # Long-Term Momentum
    df['100_day_momentum'] = df['close'].pct_change(100)
    df['200_day_momentum'] = df['close'].pct_change(200)
    df['300_day_momentum'] = df['close'].pct_change(300)
    
    # Volume-Based Momentum
    df['10_day_volume_momentum'] = df['volume'].pct_change(10)
    df['20_day_volume_momentum'] = df['volume'].pct_change(20)
    df['50_day_volume_momentum'] = df['volume'].pct_change(50)
    
    # Historical Volatility
    df['20_day_volatility'] = df['close'].pct_change().rolling(window=20).std()
    df['50_day_volatility'] = df['close'].pct_change().rolling(window=50).std()
    df['100_day_volatility'] = df['close'].pct_change().rolling(window=100).std()
    
    # Realized Volatility (Average True Range)
    df['true_range'] = df[['high', 'low']].sub(df['close'].shift(), axis=0).abs().max(axis=1)
    df['20_day_atr'] = df['true_range'].rolling(window=20).mean()
    df['50_day_atr'] = df['true_range'].rolling(window=50).mean()
    df['100_day_atr'] = df['true_range'].rolling(window=100).mean()
    
    # Moving Averages
    df['50_day_ma'] = df['close'].rolling(window=50).mean()
    df['200_day_ma'] = df['close'].rolling(window=200).mean()
    df['50_200_day_crossover'] = df['50_day_ma'] - df['200_day_ma']
    
    # Price Position Relative to Moving Averages
    df['distance_50_ma'] = (df['close'] - df['50_day_ma']) / df['50_day_ma']
    df['distance_200_ma'] = (df['close'] - df['200_day_ma']) / df['200_day_ma']
    
    # RSI (Relative Strength Index)
    def compute_rsi(prices, window):
        delta = prices.diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = up.rolling(window).mean()
        roll_down = down.abs().rolling(window).mean()
        rs = roll_up / roll_down
        return 100.0 - (100.0 / (1.0 + rs))
