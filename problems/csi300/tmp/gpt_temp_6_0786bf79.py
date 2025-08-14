import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 14-Day Volume-Weighted Intraday Return
    intraday_returns = df['close'] - df['open']
    intraday_returns_vol_weighted = (intraday_returns * df['volume']).rolling(window=14).mean()

    # Compute Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']

    # Compute Close-to-Open Return
    close_to_open_return = (df['close'] / df['open'].shift(1)) - 1

    # Combine to Form Intraday Reversal
    intraday_reversal = intraday_high_low_spread * close_to_open_return

    # Incorporate Volume Influence
    n_day_avg_volume = df['volume'].rolling(window=14).mean()
    intraday_volume_impact = df['volume'] / n_day_avg_volume
    weighted_intraday_reversal = intraday_reversal * intraday_volume_impact

    # Incorporate Amount Impact
    amount_impact = df['amount'] / n_day_avg_volume
    combined_volume_amount_impact = intraday_volume_impact + amount_impact

    # Adjust for Extreme Movement
    up_days = (df['close'] > df['open'])
    down_days = (df['open'] > df['close'])
    extreme_movement_adjustment = (df['high'] - df['low']) * (up_days - down_days)

    # Calculate Daily Log Returns
    daily_log_returns = np.log(df['close'] / df['close'].shift(1))

    # Compute Realized Volatility (20-day standard deviation of log returns)
    realized_volatility = daily_log_returns.rolling(window=20).std()

    # Calculate Intraday Volatility
    high_low_diff = df['high'] - df['low']
    intraday_volatility = high_low_diff.rolling(window=20).sum()

    # Adjust Close-to-Close Return by Intraday Volatility
    close_to_close_return = (df['close'] / df['close'].shift(1)) - 1
    adjusted_close_to_close_return = close_to_close_return / intraday_volatility

    # Enhance with Volume-Weighted High-Low Difference
    volume_weighted_high_low_diff = (df['high'] - df['low']) * df['volume']

    # Combine Adjusted Return and Volume-Weighted Intraday Return
    combined_factor = adjusted_close_to_close_return + volume_weighted_high_low_diff

    # Integrate All Factors
    integrated_factor = (combined_factor + weighted_intraday_reversal) * combined_volume_amount_impact - extreme_movement_adjustment

    return integrated_factor
