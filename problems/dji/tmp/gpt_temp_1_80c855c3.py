import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum
    recent_close = df['close'].iloc[-1]
    close_10_days_ago = df['close'].iloc[-11]
    price_momentum = recent_close - close_10_days_ago

    # Confirm with Volume Increase
    recent_volume = df['volume'].iloc[-1]
    volume_10_days_ago = df['volume'].iloc[-11]
    volume_ratio = recent_volume / volume_10_days_ago
    if volume_ratio <= 1:
        return pd.Series(0, index=df.index)

    # Calculate Exponential Moving Average (EMA) of Close Prices
    short_term_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_term_ema = df['close'].ewm(span=26, adjust=False).mean()

    # Calculate the EMA Difference
    ema_difference = short_term_ema - long_term_ema
    trend_indicator = ema_difference.iloc[-1]

    # Calculate Volume Rate of Change (VROC)
    vroc = df['volume'].pct_change(periods=14)
    vroc_value = vroc.iloc[-1]

    # Combine All Indicators
    ema_trend_product = trend_indicator * vroc_value
    if price_momentum > 0 and trend_indicator > 0:
        factor = ema_trend_product
    else:
        factor = 0

    return pd.Series(factor, index=[df.index[-1]])
