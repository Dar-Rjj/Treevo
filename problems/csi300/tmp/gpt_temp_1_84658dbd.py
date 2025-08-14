import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Daily Gain or Loss
    df['close_diff'] = df['close'] - df['close'].shift(1)
    df['daily_gain_loss'] = np.sign(df['close_diff'])

    # Adjust Gain/Loss by Volume and Price Volatility
    df['adjusted_gain_loss'] = df['daily_gain_loss'] * df['volume']
    
    # Calculate Price Volatility
    df['price_volatility'] = df['close'].rolling(window=5).std()
    df['adjusted_gain_loss'] = df['adjusted_gain_loss'] / df['price_volatility']

    # Cumulate Adjusted Value Over Window
    df['cumulative_adjusted_value'] = df['adjusted_gain_loss'].rolling(window=5).sum()

    # Calculate Exponential Moving Average (EMA) of Close Prices
    df['short_ema'] = df['close'].ewm(span=5, adjust=False).mean()
    df['long_ema'] = df['close'].ewm(span=20, adjust=False).mean()

    # Compute Momentum Difference
    df['momentum_difference'] = (df['long_ema'] - df['short_ema']).abs()

    # Calculate Volume Weighted Average Price (VWAP)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    # Integrate Cumulated Adjusted Value and Adjusted Momentum
    df['integrated_value'] = df['cumulative_adjusted_value'] * df['momentum_difference'] * df['vwap']

    # Incorporate High and Low Prices
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x[0] - x[1], 
                                                              abs(x[0] - df['close'].shift(1)), 
                                                              abs(x[1] - df['close'].shift(1))), axis=1)

    # Adjust Alpha Factor by True Range
    df['alpha_factor'] = df['integrated_value'] / df['true_range']

    return df['alpha_factor']
