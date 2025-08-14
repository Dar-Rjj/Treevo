import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['close'] * df['volume']
    df['vwap'] = df.groupby(df.index.date)['total_dollar_value'].cumsum() / df.groupby(df.index.date)['total_volume'].cumsum()

    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']

    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].cumsum()

    # Integrate Short-Term Momentum
    short_term_momentum_period = 5
    df['short_term_momentum'] = df['close'] - df['close'].shift(short_term_momentum_period)
    df['weighted_short_term_momentum'] = df['short_term_momentum'] * 0.3  # Weighting factor

    # Integrate Medium-Term Momentum
    medium_term_momentum_period = 10
    df['medium_term_momentum'] = df['close'] - df['close'].shift(medium_term_momentum_period)
    df['weighted_medium_term_momentum'] = df['medium_term_momentum'] * 0.2  # Weighting factor

    # Integrate Adaptive Exponential Moving Average (EMA)
    def adaptive_ema(data, span=20, alpha=0.2):
        ema = data.ewm(span=span, adjust=False).mean()
        return ema

    df['ema'] = adaptive_ema(df['close'])
    df['adaptive_ema'] = df['ema'] + df['close'].diff().fillna(0) * 0.1  # Adjust EMA based on recent price changes

    # Integrate Intraday Trends
    df['intraday_range'] = df['high'] - df['low']
    df['avg_intraday_range'] = df['intraday_range'].rolling(window=20).mean()
    df['intraday_range_deviation'] = df['intraday_range'] - df['avg_intraday_range']
    df['weighted_intraday_range_deviation'] = df['intraday_range_deviation'] * 0.1  # Weighting factor

    # Final Alpha Factor
    df['alpha_factor'] = (
        df['cumulative_vwap_deviation'] +
        df['weighted_short_term_momentum'] +
        df['weighted_medium_term_momentum'] +
        (df['close'] - df['adaptive_ema']) * 0.4 +  # Incorporate Adaptive EMA
        df['weighted_intraday_range_deviation']
    )

    return df['alpha_factor']
