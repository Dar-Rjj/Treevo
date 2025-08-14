import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the volume-weighted average price (VWAP) with an adaptive window
    vwap_window = 7 + df['volume'].rolling(window=14).std().fillna(0).astype(int)
    vwap = (df['amount'] / df['volume']).rolling(window=vwap_window, min_periods=1).mean()
    
    # Calculate the exponential moving average (EMA) of the close price with a dynamic window
    ema_close = df['close'].ewm(span=7+df['close'].rolling(window=14).std().fillna(0).astype(int), adjust=False).mean()

    # Calculate the relative strength index (RSI) with a dynamic window
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14+df['close'].rolling(window=14).std().fillna(0).astype(int)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14+df['close'].rolling(window=14).std().fillna(0).astype(int)).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate the logarithmic returns over a 5-day period
    log_returns = np.log(df['close'] / df['close'].shift(5))

    # Dynamic volatility as the standard deviation of log returns
    dynamic_volatility = log_returns.rolling(window=14).std()

    # Order book imbalance as a sentiment indicator
    order_book_imbalance = (df['high'] - df['low']) / df['close']

    # Combine all factors multiplicatively
    factor = ((df['close'] - ema_close) / vwap * rsi * log_returns * dynamic_volatility * order_book_imbalance)

    return factor
