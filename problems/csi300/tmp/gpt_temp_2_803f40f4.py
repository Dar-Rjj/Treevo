import pandas as pd
import pandas as pd

def heuristics_v2(df, lookback=10):
    # Calculate Intraday Volatility and High-Low Range
    intraday_volatility = df['high'] - df['low']
    high_low_range = df['high'] - df['low']

    # Weight by Volume-to-Price Ratio
    average_price = (df['high'] + df['low']) / 2
    volume_to_price_ratio = df['volume'] / average_price
    weighted_intraday_volatility = intraday_volatility * volume_to_price_ratio

    # Enhance with Close-to-Open Change
    close_to_open_change = df['close'] - df['open']
    enhanced_factor = weighted_intraday_volatility - close_to_open_change

    # Compute Final Factor
    close_open_spread_momentum = df['close'] - df['open']
    close_open_spread_momentum = close_open_spread_momentum.rolling(window=lookback).mean()

    final_factor = pd.Series(index=df.index)
    for date in df.index:
        if close_open_spread_momentum.loc[date] > 0:
            final_factor.loc[date] = enhanced_factor.loc[date] / close_open_spread_momentum.loc[date]
        else:
            final_factor.loc[date] = enhanced_factor.loc[date] * close_open_spread_momentum.loc[date]

    return final_factor
