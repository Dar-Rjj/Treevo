import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum
    latest_close = df['close'].iloc[-1]
    close_20_days_ago = df['close'].iloc[-20]
    price_momentum = latest_close - close_20_days_ago
    trend_direction = 1 if price_momentum > 0 else -1

    # Measure Daily Price Range
    daily_range = df['high'] - df['low']

    # Compute Average Daily Range over 20 Days
    average_daily_range = daily_range.rolling(window=20).mean().iloc[-1]

    # Compute Expansion Ratio
    most_recent_daily_range = daily_range.iloc[-1]
    expansion_ratio = most_recent_daily_range / average_daily_range

    # Adjust Momentum by Expansion Ratio
    adjusted_momentum = price_momentum * expansion_ratio

    # Incorporate Volume Trends
    # Calculate 20-Day Average Volume
    average_volume = df['volume'].rolling(window=20).mean().iloc[-1]

    # Calculate Volume Adjustment Factor
    most_recent_volume = df['volume'].iloc[-1]
    volume_adjustment_factor = most_recent_volume / average_volume

    # Adjust Momentum by Volume Adjustment Factor
    adjusted_momentum *= volume_adjustment_factor

    # Incorporate Volatility
    # Calculate Standard Deviation of Close Price over 20 Days
    close_std_20_days = df['close'].rolling(window=20).std().iloc[-1]

    # Normalize Volatility
    average_close_price = df['close'].rolling(window=20).mean().iloc[-1]
    normalized_volatility = close_std_20_days / average_close_price

    # Adjust Momentum by Normalized Volatility
    final_alpha_factor = adjusted_momentum * normalized_volatility

    return pd.Series(final_alpha_factor, index=[df.index[-1]])
