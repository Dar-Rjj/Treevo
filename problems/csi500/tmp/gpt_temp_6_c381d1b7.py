import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=10, short_span=12, long_span=26):
    # Calculate High-Low Spread and Intraday Move
    intraday_move = df['high'] - df['low']
    avg_close = df['close'].rolling(window=n).mean()
    scaled_high_low_spread = (df['high'] - df['low']) / avg_close

    # Calculate Close Momentum
    close_momentum = df['close'].apply(lambda x: sum([x - df['close'].shift(i) for i in range(1, n)]))

    # Incorporate Volume Weighting and Trend Following Signal
    volume_adjustment = df['volume'] / df['close']
    ema_short = df['close'].ewm(span=short_span, adjust=False).mean()
    ema_long = df['close'].ewm(span=long_span, adjust=False).mean()
    momentum_signal = ema_short - ema_long
    weighted_intraday_move = intraday_move * volume_adjustment * momentum_signal

    # Calculate Intraday Return
    intraday_return = (df['high'] - df['low']) / df['open']

    # Calculate Overlap Period Return
    overlap_period_return = (df['close'] - df['open'].shift(1)) / df['open'].shift(1)

    # Volume Growth
    volume_growth = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)

    # Combine Intraday Return and Overlap Period Return
    combined_weighted_returns = intraday_return * volume_growth + overlap_period_return * (1 - volume_growth)

    # Combine Components
    final_alpha_factor = (scaled_high_low_spread * close_momentum) + combined_weighted_returns + weighted_intraday_move

    return final_alpha_factor
