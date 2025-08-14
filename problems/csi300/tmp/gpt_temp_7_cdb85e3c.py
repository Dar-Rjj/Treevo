import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 12-day Average Price Range
    df['price_range'] = df['high'] - df['low']
    df['avg_price_range_12'] = df['price_range'].rolling(window=12).mean()

    # Compute Volume-Adjusted Momentum
    df['momentum'] = (df['close'] - df['close'].shift(12)) / df['close'].shift(12)
    df['volume_adjusted_momentum'] = df['momentum'] * df['volume']

    # Calculate Combined Intraday and Opening Gap
    df['intraday_high_low_spread_vol_weighted'] = (df['high'] - df['low']) * df['volume']
    df['opening_gap_vol_adjusted'] = (df['open'] - df['close'].shift(1)) * df['volume']
    df['combined_value'] = df['intraday_high_low_spread_vol_weighted'] + df['opening_gap_vol_adjusted']

    # Short-Term and Long-Term EMAs of Combined Value
    df['short_term_ema_12'] = df['combined_value'].ewm(span=12, adjust=False).mean()
    df['long_term_ema_30'] = df['combined_value'].ewm(span=30, adjust=False).mean()

    # Calculate Divergence
    df['divergence'] = df['long_term_ema_30'] - df['short_term_ema_12']

    # Apply Sign Function to Divergence
    df['divergence_sign'] = df['divergence'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # Incorporate Trading Volume Trend
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_trend'] = df['volume_change'].apply(lambda x: 1 if x > 0 else -1)

    # Adjust Momentum by Volume Trend
    df['adjusted_momentum'] = df['volume_adjusted_momentum'] * (1 + 0.5 * df['volume_trend'])

    # Incorporate Open-Close Trend
    df['open_close_trend'] = df['close'] - df['open']
    df['factor'] = df['adjusted_momentum'] / df['avg_price_range_12']
    df['factor'] = df['factor'] * (1.2 if df['open_close_trend'] > 0 else 0.8)

    # Combine Metrics
    df['factor'] = (df['adjusted_momentum'] / df['avg_price_range_12']) * df['divergence_sign']

    # Calculate Volume-Weighted Return Volatility
    df['daily_returns'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['return_volatility_20'] = df['daily_returns'].rolling(window=20).std() * df['volume']
    df['factor'] = df['factor'] / df['return_volatility_20']

    # Final Alpha Factor
    df['final_alpha_factor'] = df['factor'].rolling(window=7).mean()

    return df['final_alpha_factor']
