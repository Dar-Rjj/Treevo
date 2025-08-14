def heuristics_v2(df):
    def calculate_factors(data):
        data['log_price_change'] = (data['close'] / data['close'].shift(1)).apply(np.log)
        data['log_volume_change'] = (data['volume'] / data['volume'].shift(1)).apply(np.log)
        data['volatility'] = data['high'] - data['low']
        data['momentum'] = data['close'] - data['open']
        # Weighted Moving Average of Momentum with a 5-day window
        data['wma_momentum'] = data['momentum'].rolling(window=5).mean()
        return (data['log_price_change'] + data['log_volume_change'] + data['volatility'] + data['wma_momentum']) / 4
    heuristics_matrix = calculate_factors(df)
    return heuristics_matrix
