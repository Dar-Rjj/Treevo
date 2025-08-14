import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday High-to-Low Range
    daily_range = df['high'] - df['low']
    normalized_range = daily_range / df['open']

    # Open to Close Momentum
    open_to_close_return = (df['close'] - df['open']) / df['open']
    ema_open_to_close = open_to_close_return.ewm(span=5, adjust=False).mean()

    # Volume-Weighted Open-to-Close Return
    volume_weighted_return = (df['close'] - df['open']) * df['volume']
    ema_volume_weighted_return = volume_weighted_return.ewm(span=5, adjust=False).mean()

    # Price-Volume Trend Indicator
    daily_price_change = df['close'].diff(1)
    price_volume_trend = (daily_price_change * df['volume']).rolling(window=30).sum()

    # Volume-Adjusted Intraday Movement
    intraday_movement = df['close'] - df['open']
    avg_20_day_volume = df['volume'].rolling(window=20).mean()
    volume_adjusted_movement = intraday_movement / avg_20_day_volume

    # Combined Momentum and Volatility Factor
    combined_momentum_volatility = 0.4 * normalized_range + 0.3 * ema_open_to_close + 0.3 * ema_volume_weighted_return
    combined_momentum_volatility = combined_monomentum_volatility.ewm(alpha=0.2, adjust=False).mean()

    # Volume-Sensitive Momentum Factor
    volume_sensitive_momentum = 0.4 * price_volume_trend + 0.3 * ema_volume_weighted_return + 0.3 * volume_adjusted_movement
    volume_sensitive_momentum = volume_sensitive_momentum.ewm(alpha=0.2, adjust=False).mean()

    # Final Alpha Factor
    final_alpha_factor = 0.6 * combined_momentum_volatility + 0.4 * volume_sensitive_momentum
    final_alpha_factor = final_alpha_factor.ewm(alpha=0.2, adjust=False).mean()

    return final_alpha_factor
