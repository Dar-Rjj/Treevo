import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Enhanced Momentum
    df['high_low_delta'] = df['high'] - df['low']
    df['close_open_delta'] = df['close'] - df['open']
    df['enhanced_momentum_score'] = df['high_low_delta'] + df['close_open_delta']

    # Weighted by Previous Day's Momentum Score with Exponential Decay
    decay_factor = 0.95
    df['prev_momentum_score'] = df['enhanced_momentum_score'].shift(1)
    df['enhanced_momentum_score'] = (df['high_low_delta'] + df['close_open_delta']) * (1 - decay_factor) + df['prev_momentum_score'] * decay_factor

    # Confirm Momentum with Volume Surge and Adjustment
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    volume_threshold = df['volume'].mean() * 0.2  # Set a minimum threshold for volume increase
    momentum_threshold = df['enhanced_momentum_score'].abs().mean() * 0.2  # Set a minimum threshold for momentum score
    df['volume_weighting'] = df['volume_change'] * df['enhanced_momentum_score']
    df['significant_signal'] = (df['volume_change'] > volume_threshold) & (df['enhanced_momentum_score'].abs() > momentum_threshold)

    # Calculate Volume Adjustment Factor
    long_term_avg_volume = df['volume'].rolling(window=20).mean()
    short_term_avg_volume = df['volume'].rolling(window=5).mean()
    df['volume_adjustment_factor'] = short_term_avg_volume / long_term_avg_volume

    # Integrate Price and Volume Changes with Efficiency
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['price_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'])

    # Cumulative Enhanced Momentum Indicator
    df['cumulative_momentum'] = (df['price_change'] * df['volume_change'] * df['price_efficiency']).rolling(window=10).sum()

    # Smooth the Indicator
    df['sma_cumulative_momentum'] = df['cumulative_momentum'].rolling(window=10).mean()

    # Enhance with Trend Analysis
    long_term_trend = df['close'].rolling(window=200).mean()
    short_term_trend = df['close'].rolling(window=50).mean()
    trend_amplifier = 1.2 if short_term_trend > long_term_trend else 0.8
    df['adjusted_cumulative_momentum'] = df['sma_cumulative_momentum'] * trend_amplifier

    # Smooth the Adjusted Indicator
    df['final_factor'] = df['adjusted_cumulative_momentum'].rolling(window=10).mean()

    # Incorporate Volume Volatility and Price Oscillation
    long_term_vol_std = df['volume'].rolling(window=20).std()
    short_term_vol_std = df['volume'].rolling(window=5).std()
    vol_volatility_ratio = short_term_vol_std / long_term_vol_std

    long_term_price_std = df['close'].rolling(window=20).std()
    short_term_price_std = df['close'].rolling(window=5).std()
    price_oscillation_ratio = short_term_price_std / long_term_price_std

    df['final_factor'] = df['final_factor'] * (vol_volatility_ratio + price_oscillation_ratio)

    return df['final_factor'].dropna()
