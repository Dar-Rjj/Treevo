import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Weighted Breakout Intensity
    breakout_intensity = (df['High'] - df['Low']) / df['Close']
    breakout_intensity[breakout_intensity < 0] = 0
    volume_weighted_breakout = breakout_intensity * df['Volume']

    # Volume-Adjusted Momentum (VAM)
    close_sma = df['Close'].rolling(window=20).mean()
    momentum = close_sma.diff(periods=5)
    avg_volume = df['Volume'].rolling(window=5).mean()
    volume_adjusted_momentum = momentum * df['Volume'] / avg_volume

    # Price Volatility Indicator (PVI)
    true_range = df[['High', 'Low', 'Close']].join(df['Close'].shift(1)).apply(
        lambda x: max(x['High'] - x['Low'], abs(x['High'] - x['Close']), abs(x['Low'] - x['Close'])), axis=1)
    atr = true_range.rolling(window=14).mean()
    pvi = true_range / atr
    pvi_smoothed = pvi.ewm(span=7).mean()

    # Volume Increase Ratio
    volume_increase_ratio = df['Volume'] / df['Volume'].shift(1)

    # Close Price Momentum
    close_momentum = df['Close'] - df['Close'].rolling(window=5).mean()

    # Price Range
    price_range = df['High'] - df['Low']

    # Combine Factors
    combined_factor = (volume_weighted_breakout * volume_adjusted_momentum) / pvi_smoothed
    combined_factor = combined_factor * volume_increase_ratio * close_momentum / price_range
    adjusted_factor = combined_factor - combined_factor.rolling(window=30).mean()

    return adjusted_factor
