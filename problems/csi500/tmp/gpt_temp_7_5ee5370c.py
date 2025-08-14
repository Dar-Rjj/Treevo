import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:

    # Calculate the volume-weighted average price (VWAP) with a dynamic window
    vwap = (df['amount'] / df['volume']).rolling(window=7, min_periods=1).mean()

    # Calculate the adaptive exponential moving average (EMA) of the close price with dynamic windows
    ema_close = df['close'].ewm(span=7, adjust=False).mean()
    ema_close_50 = df['close'].ewm(span=50, adjust=False).mean()

    # Calculate the relative strength index (RSI) with a dynamic window
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-7)
    rsi = 100 - (100 / (1 + rs))

    # Calculate the logarithmic returns over a 5-day period
