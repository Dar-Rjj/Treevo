import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    intraday_high_low_spread = df['High'] - df['Low']

    # Compute Previous Day's Close-to-Open Return
    prev_close = df['Close'].shift(1)
    close_to_open_return = df['Open'] - prev_close

    # Compute Enhanced Intraday Momentum
    enhanced_intraday_momentum = (intraday_high_low_spread + close_to_open_return) / 2

    # Compute Daily VWAP
    prices = (df['High'] + df['Low'] + df['Close'] + df['Open']) / 4
    vwap = (prices * df['Volume']).cumsum() / df['Volume'].cumsum()

    # Integrate Intraday Momentum and VWAP
    integrated_factor = vwap - intraday_high_low_spread
    weighted_integrated_factor = integrated_factor * df['Volume']

    # Calculate Volume Trend
    volume_3d_ma = df['Volume'].rolling(window=3).mean()
    volume_trend = df['Volume'] - volume_3d_ma

    # Incorporate Sentiment Score (Assuming a 'Sentiment' column is available in the DataFrame)
    sentiment_score = df['Sentiment']

    # Final Alpha Factor
    alpha_factor = weighted_integrated_factor + volume_trend + sentiment_score

    # Apply Exponential Moving Average (EMA) over the last 3 days
    alpha_factor_ema = alpha_factor.ewm(span=3, adjust=False).mean()

    return alpha_factor_ema
