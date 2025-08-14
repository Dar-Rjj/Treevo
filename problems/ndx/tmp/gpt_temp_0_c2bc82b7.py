def heuristics_v2(df):
    # Calculating a simple moving average for closing prices over a 10-day period
    sma_10 = df['close'].rolling(window=10).mean()
    # Calculating the difference between current close price and its 10-day SMA to identify trends
    trend_indicator = df['close'] - sma_10
    # Considering the relationship between volume and price changes as a measure of strength
    volume_adjusted_trend = trend_indicator * df['volume']
    # Applying a 5-day lagged moving average to smooth out the noise
    smoothed_heuristic = volume_adjusted_trend.rolling(window=5).mean()
    return heuristics_matrix
