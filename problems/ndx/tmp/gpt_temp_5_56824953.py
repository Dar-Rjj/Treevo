def heuristics_v2(df):
    # Price Momentum
    df['5d_price_momentum'] = df['close'].pct_change(5)
    df['20d_price_momentum'] = df['close'].pct_change(20)
    df['60d_price_momentum'] = df['close'].pct_change(60)
    
    # Volume Momentum
    df['20d_avg_volume'] = df['volume'].rolling(window=20).mean()
    df['volume_momentum'] = df['volume'] / df['20d_avg_volume']
    
    # Historical Volatility
    df['10d_volatility'] = df['close'].pct_change().rolling(window=10).std()
    df['30d_volatility'] = df['close'].pct_change().rolling(window=30).std()
    df['60d_volatility'] = df['close'].pct_change().rolling(window=60).std()
    df['120d_volatility'] = df['close'].pct_change().rolling(window=120).std()
    
    # Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']) / df['open']
    df['5d_intraday_volatility'] = df['intraday_volatility'].rolling(window=5).mean()
    df['20d_intraday_volatility'] = df['intraday_volatility'].rolling(window=20).mean()
    
    # Trend Strength Indicators
    def ADX(df, period=14):
        df['high_low_diff'] = df['high'] - df['low']
        df['high_close_diff'] = abs(df['high'] - df['close'].shift(1))
        df['low_close_diff'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['high_low_diff', 'high_close_diff', 'low_close_diff']].max(axis=1)
        
        df['pos_dm'] = 0
        df['neg_dm'] = 0
        df.loc[df['high'] > df['high'].shift(1), 'pos_dm'] = df['high'] - df['high'].shift(1)
        df.loc[df['low'].shift(1) > df['low'], 'neg_dm'] = df['low'].shift(1) - df['low']
        
        df['pos_di'] = 100 * (df['pos_dm'].rolling(window=period).sum() / df['true_range'].rolling(window=period).sum())
        df['neg_di'] = 100 * (df['neg_dm'].rolling(window=period).sum() / df['true_range'].rolling(window=period).sum())
        df['adx'] = 100 * (abs(df['pos_di'] - df['neg_di']) / (df['pos_di'] + df['neg_di']))
        
        return df['adx']
