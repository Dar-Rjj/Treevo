import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Exponential Moving Average (EMA) Momentum
    df['1d_return'] = df['close'].pct_change()
    df['5d_return'] = df['close'].pct_change(5)
    df['20d_return'] = df['close'].pct_change(20)
    df['ema_momentum'] = df[['1d_return', '5d_return', '20d_return']].apply(lambda x: (x * 0.3).ewm(alpha=0.3).mean().sum(axis=1), axis=1)

    # Weighted Average of Price Gaps
    df['price_gap'] = abs(df['close'] - df['open'])
    df['5d_volume_ma'] = df['volume'].rolling(window=5).mean()
    df['volume_growth'] = df['volume'] / df['volume'].shift(1)
    df['volume_trend'] = df['5d_volume_ma'] * df['volume_growth']
    df['weighted_price_gaps'] = df['price_gap'] * df['volume_trend']

    # Adjust for Volume
    df['volume_pct_change'] = df['volume'].pct_change()
    df['adjusted_weighted_price_gaps'] = df['weighted_price_gaps'] * df['volume_pct_change']

    # Adaptive High-to-Low Range and Momentum
    df['high_low_range'] = df['high'] - df['low']
    df['combined_value'] = df['high_low_range'] * 100 + df['weighted_price_gaps']
    df['smoothed_combined_value'] = df['combined_value'].ewm(span=5).mean()
    df['momentum'] = df['smoothed_combined_value'] - df['smoothed_combined_value'].shift(5)

    # Calculate Rolling High-Low Differential
    df['20d_rolling_high'] = df['high'].rolling(window=20).max()
    df['20d_rolling_low'] = df['low'].rolling(window=20).min()
    df['rolling_high_low_diff'] = df['20d_rolling_high'] - df['20d_rolling_low']

    # Weight by Volume Volatility
    df['volume_volatility'] = df['volume'].rolling(window=10).std()
    df['weighted_rolling_high_low_diff'] = df['rolling_high_low_diff'] * df['volume_volatility']

    # Aggregate the Signals
    alpha_factor = (
        df['weighted_rolling_high_low_diff'] -
        df['adjusted_weighted_price_gaps'] +
        df['momentum'] -
        df['ema_momentum']
    )

    return alpha_factor
