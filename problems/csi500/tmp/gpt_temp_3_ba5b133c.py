import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive window for VWAP, RSI, and EMA
    adaptive_window = 7 + (df['close'].rolling(window=30).std() / df['close'].rolling(window=30).mean()) * 14

    # Calculate the volume-weighted average price (VWAP) with an adaptive window
    vwap = (df['amount'] / df['volume']).rolling(window=adaptive_window.astype(int)).mean()

    # Calculate the exponential moving average (EMA) of the close price with an adaptive window
    ema_close = df['close'].ewm(span=adaptive_window.astype(int), adjust=False).mean()

    # Calculate the relative strength index (RSI) with an adaptive window
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=adaptive_window.astype(int)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=adaptive_window.astype(int)).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate the logarithmic returns over a 5-day period
    log_returns = np.log(df['close'] / df['close'].shift(5))

    # Dynamic volatility over a 30-day window
    dynamic_volatility = df['close'].pct_change().rolling(window=30).std()

    # Combine multiplicatively and normalize with ratios
    factor = ((df['close'] - ema_close) / vwap) * (rsi / 50) * (log_returns / dynamic_volatility)

    return factor
