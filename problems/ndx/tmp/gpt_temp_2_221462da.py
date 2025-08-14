import pandas as pd

def heuristics_v2(df):
    # Calculate short and long term weighted moving averages for the closing price
    weights_short = np.arange(1, 11)
    wma_short_term = df['close'].rolling(window=10).apply(lambda prices: np.dot(prices, weights_short)/weights_short.sum(), raw=True)
    weights_long = np.arange(1, 61)
    wma_long_term = df['close'].rolling(window=60).apply(lambda prices: np.dot(prices, weights_long)/weights_long.sum(), raw=True)

    # Volume-weighted median price
    vw_median_price = (df[['open', 'high', 'low', 'close']].multiply(df['volume'], axis=0)).median(axis=1) / df['volume']

    delta = vw_median_price.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Integrate the WMA difference and the modified RSI
    wma_diff = wma_short_term - wma_long_term
    heuristics_matrix = (wma_diff + rsi) / 2

    return heuristics_matrix
