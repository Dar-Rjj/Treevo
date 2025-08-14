import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Volatility and High-Low Range
    intraday_volatility = df['high'] - df['low']
    high_low_range = df['high'] - df['low']

    # Weight by Volume-to-Price Ratio
    average_price = (df['high'] + df['low']) / 2
    volume_to_price_ratio = df['volume'] / average_price
    weighted_intraday_volatility = intraday_volatility * volume_to_price_ratio

    # Enhance with Close-to-Open Change
    close_open_change = df['close'] / df['open'] - 1
    enhanced_weighted_volatility = weighted_intraday_volatility - close_open_change

    # Calculate Momentum
    lookback_period = 10
    high_low_momentum = df['high'].rolling(window=lookback_period).mean() - df['low'].rolling(window=lookback_period).mean()
    close_open_momentum = df['close'].rolling(window=lookback_period).mean() - df['open'].rolling(window=lookback_period).mean()

    # Compute Final Factor
    combined_factor = enhanced_weighted_volatility
    final_factor = combined_factor.copy()

    for i in range(len(final_factor)):
        if close_open_momentum.iloc[i] > 0:
            final_factor.iloc[i] /= close_open_momentum.iloc[i]
        else:
            final_factor.iloc[i] *= close_open_momentum.iloc[i]

    return final_factor
