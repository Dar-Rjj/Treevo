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
    close_open_change = (df['close'] - df['open']) / df['open']
    enhanced_volatility = weighted_intraday_volatility - close_open_change

    # Calculate Momentum
    lookback_period = 10
    high_low_momentum = high_low_range.rolling(window=lookback_period).mean()
    close_open_momentum = (df['close'] - df['open']).rolling(window=lookback_period).mean()

    # Compute Final Factor
    final_factor = enhanced_volatility
    for i in range(lookback_period, len(df)):
        if close_open_momentum[i] > 0:
            final_factor[i] = final_factor[i] / close_open_momentum[i]
        else:
            final_factor[i] = final_factor[i] * close_open_momentum[i]

    return pd.Series(final_factor, index=df.index)
