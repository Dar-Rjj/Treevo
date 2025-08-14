import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the volume-weighted average price (VWAP) with an adaptive window
    vwap = (df['amount'] / df['volume']).rolling(window=int(7 + df['close'].rolling(window=7).std() * 10)).mean()

    # Calculate the exponential moving average (EMA) of the close price with an adaptive window
    ema_close = df['close'].ewm(span=int(7 + df['close'].rolling(window=7).std() * 10), adjust=False).mean()

    # Calculate the relative strength index (RSI) with an adaptive window
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=int(14 + df['close'].rolling(window=7).std() * 10)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=int(14 + df['close'].rolling(window=7).std() * 10)).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate the logarithmic returns over a dynamic period
    log_returns = np.log(df['close'] / df['close'].shift(int(5 + df['close'].rolling(window=7).std() * 10)))

    # Calculate sentiment as the difference between the high and low prices, normalized by the VWAP
    sentiment = (df['high'] - df['low']) / vwap

    # Calculate the factor combining EMA, RSI, logarithmic returns, and sentiment
    factor = ((df['close'] - ema_close) * rsi * log_returns * sentiment)

    return factor
