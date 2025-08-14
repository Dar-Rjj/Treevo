import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Returns
    df['daily_return'] = df['close'].pct_change()

    # Sum 5-Day Returns
    df['5_day_return_sum'] = df['daily_return'].rolling(window=5).sum()

    # Adjust Factor Based on Volume
    df['volume_change'] = df['volume'].diff()
    df['confirmed_price_momentum'] = df.apply(lambda row: row['5_day_return_sum'] if row['volume_change'] > 0 else 0, axis=1)

    # Calculate Intraday Price Movement
    df['intraday_range'] = df['high'] - df['low']

    # Calculate Close to Open Difference
    df['co_difference'] = df['close'] - df['open']

    # Calculate High-Low Momentum
    df['high_low_momentum'] = (df['high'] - df['low']).rolling(window=5).mean()

    # Calculate Volume Scaled Factor
    df['volume_scaled_co'] = df['volume'] * df['co_difference']

    # Calculate Intraday Volatility
    df['5_day_intraday_volatility'] = df['intraday_range'].abs().rolling(window=5).sum()

    # Compute Intraday Stability
    df['intraday_stability'] = 1 / (df['5_day_intraday_volatility'] / df['intraday_range'])

    # Calculate Volume Trend
    df['volume_trend'] = df['volume'].rolling(window=5).mean().apply(lambda x: 1 if x < df['volume'] else -1)

    # Combine Factors
    df['combined_alpha'] = (
        0.3 * df['confirmed_price_momentum'] +
        0.2 * df['high_low_momentum'] +
        0.2 * df['volume_scaled_co'] +
        0.15 * df['intraday_stability'] +
        0.15 * df['volume_trend']
    )

    # Finalize Alpha
    df['long_term_smoothed_return'] = df['daily_return'].rolling(window=20).mean()
    df['final_alpha_factor'] = df['combined_alpha'] - df['long_term_smoothed_return']

    return df['final_alpha_factor'].dropna()
