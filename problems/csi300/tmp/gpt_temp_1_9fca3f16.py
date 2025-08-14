import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Volatility and High-Low Range
    intraday_volatility = df['high'] - df['low']
    high_low_range = df['high'] - df['low']

    # Weight by Volume-to-Price Ratio
    avg_price = (df['high'] + df['low']) / 2
    volume_to_price_ratio = df['volume'] / avg_price
    weighted_intraday_volatility = intraday_volatility * volume_to_price_ratio

    # Enhance with Close-to-Open Change
    close_open_diff = df['close'] - df['open']
    relative_change = close_open_diff / df['open']
    enhanced_factor = weighted_intraday_volatility - relative_change

    # Calculate Momentum
    lookback_period = 10
    momentum_high_low = high_low_range.rolling(window=lookback_period).mean()
    momentum_close_open = close_open_diff.rolling(window=lookback_period).mean()

    # Compute Final Factor
    combined_factor = enhanced_factor
    adjusted_factor = combined_factor.copy()
    
    for i in range(lookback_period, len(df)):
        if momentum_close_open[i] > 0:
            adjusted_factor.iloc[i] = combined_factor.iloc[i] / momentum_close_open[i]
        else:
            adjusted_factor.iloc[i] = combined_factor.iloc[i] * momentum_close_open[i]

    return adjusted_factor
