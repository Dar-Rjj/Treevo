import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']

    # Calculate Daily Volume Trend
    volume_trend = df['volume'] - df['volume'].rolling(window=10).mean()
    volume_trend_sign = np.sign(volume_trend)

    # Calculate Short-Term Price Trend (EMA over 10 days)
    short_term_ema = df['close'].ewm(span=10, adjust=False).mean()
    short_term_trend = df['close'] - short_term_ema
    short_term_trend_sign = np.sign(short_term_trend)

    # Calculate Medium-Term Price Trend (EMA over 30 days)
    medium_term_ema = df['close'].ewm(span=30, adjust=False).mean()
    medium_term_trend = df['close'] - medium_term_ema
    medium_term_trend_sign = np.sign(medium_term_trend)

    # Calculate Long-Term Price Trend (EMA over 50 days)
    long_term_ema = df['close'].ewm(span=50, adjust=False).mean()
    long_term_trend = df['close'] - long_term_ema
    long_term_trend_sign = np.sign(long_term_trend)

    # Calculate Dynamic Volatility
    volatility = df['close'].rolling(window=10).std()
    percentile_80 = volatility.quantile(0.80)
    percentile_60 = volatility.quantile(0.60)
    percentile_40 = volatility.quantile(0.40)
    
    volatility_score = np.where(volatility > percentile_80, 2.5,
                                np.where(volatility > percentile_60, 1.5,
                                         np.where(volatility > percentile_40, 1.0, 0.5)))

    # Integrate Momentum and Relative Strength
    short_term_ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    long_term_ema_20 = df['close'].ewm(span=20, adjust=False).mean()
    relative_strength = short_term_ema_5 / long_term_ema_20
    momentum_factor = np.where(relative_strength > 1, 1.7, 0.3)

    # Combine Spread, Volume, Multi-Period Price Trends, and Volatility
    adjusted_high_low_spread = high_low_spread * (1.5 if volume_trend_sign > 0 else 0.5)
    adjusted_short_term_trend = adjusted_high_low_spread * (1.2 if short_term_trend_sign > 0 else 0.8)
    adjusted_medium_term_trend = adjusted_short_term_trend * (1.1 if medium_term_trend_sign > 0 else 0.9)
    adjusted_long_term_trend = adjusted_medium_term_trend * (1.3 if long_term_trend_sign > 0 else 0.7)
    final_alpha = adjusted_long_term_trend * volatility_score * momentum_factor

    # Consider Dynamic Market Context
    # Placeholder for market trend and sector performance
    # market_trend = 'bullish'  # or 'bearish'
    # sector_performance = 'outperforming'  # or 'underperforming'

    # Apply market trend and sector performance adjustments
    # if market_trend == 'bullish':
    #     final_alpha *= 1.5
    # elif market_trend == 'bearish':
    #     final_alpha *= 0.5

    # if sector_performance == 'outperforming':
    #     final_alpha *= 1.2
    # elif sector_performance == 'underperforming':
    #     final_alpha *= 0.8

    return final_alpha
