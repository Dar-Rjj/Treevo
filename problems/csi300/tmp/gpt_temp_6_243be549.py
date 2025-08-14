import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the 5-day and 20-day exponential moving average of closing prices to capture momentum
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Momentum factor: difference between 5-day and 20-day EMA
    df['momentum_factor'] = df['EMA_5'] - df['EMA_20']

    # Calculate the 20-day standard deviation of closing prices to capture volatility
    df['volatility_factor'] = df['close'].rolling(window=20).std()

    # Calculate the 5-day rolling sum of volume to capture liquidity
    df['liquidity_factor'] = df['volume'].rolling(window=5).sum()

    # Calculate relative strength as the ratio of current close to 20-day EMA
    df['relative_strength'] = df['close'] / df['EMA_20']

    # Calculate market breadth: difference between 5-day and 20-day moving averages of (high - low)
    df['range_5'] = (df['high'] - df['low']).rolling(window=5).mean()
    df['range_20'] = (df['high'] - df['low']).rolling(window=20).mean()
    df['market_breadth'] = df['range_5'] - df['range_20']

    # Incorporate Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['amount'] / df['volume']).rolling(window=20).mean()
    df['VWAP_deviation'] = (df['close'] - df['VWAP']) / df['VWAP']

    # Calculate logarithmic returns
