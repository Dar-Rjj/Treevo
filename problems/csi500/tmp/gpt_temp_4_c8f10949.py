import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['close'] * df['volume']
    df['vwap'] = df.groupby(df.index.date)['total_dollar_value'].transform('sum') / df.groupby(df.index.date)['total_volume'].transform('sum')

    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']

    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].groupby(df.index.date).cumsum()

    # Integrate Adaptive Momentum
    short_term_momentum_period = 5
    medium_term_momentum_period = 10
    long_term_momentum_period = 20

    # Calculate Short, Medium, and Long-Term Momentum
    df['short_term_momentum'] = df['vwap_deviation'].rolling(window=short_term_momentum_period).sum()
    df['medium_term_momentum'] = df['vwap_deviation'].rolling(window=medium_term_momentum_period).sum()
    df['long_term_momentum'] = df['vwap_deviation'].rolling(window=long_term_momentum_period).sum()

    # Combine Adaptive Momentum (using equal weights for simplicity)
    weights = [0.33, 0.33, 0.34]  # Equal weights
    df['adaptive_momentum'] = (df['short_term_momentum'] * weights[0] +
                               df['medium_term_momentum'] * weights[1] +
                               df['long_term_momentum'] * weights[2])

    # Add Weighted Momentum to Cumulative VWAP Deviation
    df['momentum_enhanced_factor'] = df['cumulative_vwap_deviation'] + df['adaptive_momentum']

    # Calculate Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['absolute_vwap_deviation'] = (df['close'] - df['vwap']).abs()
    df['intraday_volatility'] = df['high_low_range'] + df['absolute_vwap_deviation']

    # Final Alpha Factor
    df['final_alpha_factor'] = df['momentum_enhanced_factor'] + df['intraday_volatility']

    return df['final_alpha_factor']
