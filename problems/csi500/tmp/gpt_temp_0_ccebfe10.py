import pandas as pd
import pandas as pd

def heuristics_v2(df, lookback_period=20, n_days=5, m_days=10):
    # Calculate Volume-Weighted High-Low Spread
    high_low_spread = df['high'] - df['low']
    volume_weighted_high_low = (high_low_spread * df['volume']) / df['volume']

    # Compute Cumulative Volume-Weighted High-Low Spread
    cumulative_high_low_spread = (high_low_spread * df['volume']).rolling(window=lookback_period).sum()
    total_volume = df['volume'].rolling(window=lookback_period).sum()
    cumulative_volume_weighted_high_low = cumulative_high_low_spread / total_volume

    # Calculate Momentum Indicator
    current_day_high_low = volume_weighted_high_low
    past_n_days_high_low = cumulative_volume_weighted_high_low.rolling(window=n_days).mean()
    momentum = current_day_high_low - past_n_days_high_low

    # Calculate Price Momentum
    price_momentum = df['close'].pct_change(n_days)

    # Calculate Volume-Weighted Volatility
    daily_range = df['high'] - df['low']
    volatility = daily_range.rolling(window=n_days).std()
    volume_weighted_volatility = volatility * df['volume']

    # Calculate Intraday Range Growth
    today_range = df['high'] - df['low']
    prev_day_range = df[['high', 'low']].shift(1).diff(axis=1).fillna(0)
    intraday_range_growth = (today_range - prev_day_range) / prev_day_range

    # Volume Weighted Moving Average
    simple_moving_average = df['close'].rolling(window=n_days).mean()
    volume_weighted_close = df['close'] * df['volume']
    total_volume_n_days = df['volume'].rolling(window=n_days).sum()
    volume_weighted_ma = volume_weighted_close.rolling(window=n_days).sum() / total_volume_n_days

    # Combine Factors
    combined_factors = (
        price_momentum * volume_weighted_volatility +
        momentum +
        intraday_range_growth +
        volume_weighted_ma
    )

    # Calculate Volume Surge
    n_day_volume_change = df['volume'] - df['volume'].shift(n_days)
    m_day_volume_change = df['volume'] - df['volume'].shift(m_days)
    volume_surge = n_day_volume_change + m_day_volume_change

    # Apply Penalty if Volume Surge is Negative
    penalty = (volume_surge < 0) * 0.8
    final_alpha_factor = combined_factors * (1 - penalty)

    return final_alpha_factor
