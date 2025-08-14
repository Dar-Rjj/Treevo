def heuristics_v2(df):
    # Momentum Factors
    df['short_term_momentum'] = (df['close'] / df['close'].shift(1)) - 1
    df['medium_term_momentum'] = (df['close'] / df['close'].shift(20)) - 1
    df['long_term_momentum'] = (df['close'] / df['close'].shift(252)) - 1
    
    # Volatility Indicators
    df['price_range'] = df['high'] - df['low']
    
    def average_true_range(data, window=14):
        tr = data['high'] - data['low']
        atr = tr.rolling(window=window).mean()
        return atr
