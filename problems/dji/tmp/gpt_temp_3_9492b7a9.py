def heuristics_v2(df):
    def calculate_momentum(price_series, window=10):
        return price_series.pct_change(window).dropna()

    def calculate_volatility(price_series, window=10):
        return price_series.pct_change().rolling(window=window).std().dropna()
    
    def custom_indicator(price_series, vol_series, window=10):
        return (price_series / vol_series).rolling(window).mean().dropna()

    momentum = df['close'].apply(calculate_momentum)
    volatility = df['close'].apply(calculate_volatility)
    custom_ind = custom_indicator(df['close'], df['volume'])

    heuristics_matrix = (momentum + 1) * (volatility + 1) * (custom_ind + 1)
    heuristics_matrix.dropna(inplace=True)
    
    return heuristics_matrix
