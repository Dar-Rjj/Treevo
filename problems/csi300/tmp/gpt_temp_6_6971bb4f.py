import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the 30-day and 90-day simple moving average of close prices
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['sma_90'] = df['close'].rolling(window=90).mean()

    # Calculate the ratio of the 30-day to 90-day SMA, representing a medium to long-term trend
    sma_ratio = df['sma_30'] / df['sma_90']

    # Calculate the 7-day volume-weighted average price (VWAP)
    df['vwap'] = (df['close'] * df['volume']).rolling(window=7).sum() / df['volume'].rolling(window=7).sum()

    # Calculate the relative change in VWAP over the past 14 days
    vwap_change = df['vwap'].pct_change(periods=14)

    # Calculate the 30-day and 90-day standard deviation of close prices for volatility
    df['volatility_30'] = df['close'].rolling(window=30).std()
    df['volatility_90'] = df['close'].rolling(window=90).std()

    # Calculate the ratio of 30-day to 90-day volatility
    volatility_ratio = df['volatility_30'] / (df['volatility_90'] + 1e-7)

    # Calculate the 30-day and 90-day cumulative volume for liquidity
    df['liquidity_30'] = df['volume'].rolling(window=30).sum()
    df['liquidity_90'] = df['volume'].rolling(window=90).sum()

    # Calculate the ratio of 30-day to 90-day liquidity
    liquidity_ratio = df['liquidity_30'] / (df['liquidity_90'] + 1e-7)

    # Calculate the 30-day and 90-day cumulative amount for transaction activity
    df['activity_30'] = df['amount'].rolling(window=30).sum()
    df['activity_90'] = df['amount'].rolling(window=90).sum()

    # Calculate the ratio of 30-day to 90-day transaction activity
    activity_ratio = df['activity_30'] / (df['activity_90'] + 1e-7)

    # Calculate the 30-day and 90-day range of high and low prices
    df['range_30'] = df['high'].rolling(window=30).max() - df['low'].rolling(window=30).min()
    df['range_90'] = df['high'].rolling(window=90).max() - df['low'].rolling(window=90).min()

    # Calculate the ratio of 30-day to 90-day range
    range_ratio = df['range_30'] / (df['range_90'] + 1e-7)

    # Calculate the 30-day and 90-day return
