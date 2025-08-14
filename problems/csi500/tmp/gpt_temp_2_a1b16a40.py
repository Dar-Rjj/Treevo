import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the volume-weighted average price (VWAP) with a dynamic window based on 7-day rolling mean of volume
    vwap = (df['amount'] / df['volume']).rolling(window=int(df['volume'].rolling(window=7).mean())).mean()

    # Calculate the adaptive exponential moving average (EMA) of the close price, adjust span dynamically based on recent volatility
    vol = df['close'].rolling(window=7).std()
    ema_close = df['close'].ewm(span=int(7 + 3 * vol), adjust=False).mean()

    # Calculate the relative strength index (RSI) with an adaptive window size based on the 14-day average true range
    atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().fillna(0)
    rsi_window = int(14 + 5 * atr)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Logarithmic returns over a 5-day period to capture trend
    log_returns = np.log(df['close'] / df['close'].shift(5))

    # Market sentiment factor: calculate the difference between the closing and opening prices normalized by the daily range
    market_sentiment = (df['close'] - df['open']) / (df['high'] - df['low'])

    # Cross-asset correlation as a proxy for market coherence; using high and low for simplicity here
    cross_asset_corr = df['high'].corr(df['low'])

    # Incorporate VWAP, EMA, RSI, logarithmic returns, market sentiment, and cross-asset correlations into the alpha factor
    factor = ((df['close'] - ema_close) / vwap) * rsi * log_returns * market_sentiment * (1 + 0.01 * cross_asset_corr)

    return factor
