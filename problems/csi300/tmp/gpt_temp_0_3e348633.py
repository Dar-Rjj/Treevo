import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 10-day Average Price Range
    df['price_range'] = df['high'] - df['low']
    df['avg_price_range'] = df['price_range'].rolling(window=10).mean()

    # Compute Volume-Adjusted Momentum
    df['momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['volume_adjusted_momentum'] = df['momentum'] * df['volume']

    # Calculate Combined Intraday and Opening Gap
    df['intraday_high_low_spread'] = (df['high'] - df['low']) * df['volume']
    df['opening_gap'] = (df['open'] - df['close'].shift(1)) * df['volume']
    df['combined_value'] = df['intraday_high_low_spread'] + df['opening_gap']

    # Short-Term and Long-Term EMAs of Combined Value
    df['short_term_ema'] = df['combined_value'].ewm(span=12, adjust=False).mean()
    df['long_term_ema'] = df['combined_value'].ewm(span=26, adjust=False).mean()

    # Calculate Divergence
    df['divergence'] = df['long_term_ema'] - df['short_term_ema']

    # Apply Sign Function to Divergence
    df['sign_divergence'] = df['divergence'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # Combine Metrics
    df['momentum_divergence'] = df['volume_adjusted_momentum'] / df['avg_price_range'] + df['sign_divergence']

    # Incorporate Trading Volume Trend
    df['volume_change'] = df['volume'] - df['volume'].shift(10)
    df['momentum_with_volume_trend'] = df['momentum_divergence'] + df['volume_change']

    # Incorporate Open-Close Trend
    df['open_close_trend'] = df['close'] - df['open']
    df['adjusted_factor'] = df['momentum_with_volume_trend'].where(df['open_close_trend'] == 0, 
                                                                  df['momentum_with_volume_trend'] * 1.2,
                                                                  df['momentum_with_volume_trend'] * 0.8)

    # Final Alpha Factor
    df['final_alpha_factor'] = df['adjusted_factor'].rolling(window=7).mean()

    return df['final_alpha_factor']
