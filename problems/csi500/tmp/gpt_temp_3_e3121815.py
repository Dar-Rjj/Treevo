import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Short-Term Moving Average (5 days)
    short_term_window = 5
    df['short_term_ma'] = df['close'].rolling(window=short_term_window).mean()

    # Calculate Long-Term Moving Average (20 days)
    long_term_window = 20
    df['long_term_ma'] = df['close'].rolling(window=long_term_window).mean()

    # Compute Price Oscillator
    df['price_oscillator'] = df['short_term_ma'] - df['long_term_ma']

    # Calculate Volume-Weighted Adjustment
    recent_volume_window = 5
    historical_volume_window = 20
    df['recent_volume'] = df['volume'].rolling(window=recent_volume_window).mean()
    df['historical_volume'] = df['volume'].rolling(window=historical_volume_window).mean()
    df['volume_ratio'] = df['recent_volume'] / df['historical_volume']

    # Adjust Price Oscillator by Volume
    df['adjusted_price_oscillator'] = df['price_oscillator'] * df['volume_ratio']

    # Calculate Volatility (20 days)
    volatility_window = 20
    df['volatility'] = df['close'].rolling(window=volatility_window).std()

    # Compute Volatility-Adjusted Price Oscillator
    df['vol_adj_price_oscillator'] = df['adjusted_price_oscillator'] / df['volatility']

    # Apply Leverage Moving Averages (10 days)
    lma_window = 10
    df['lma'] = df['close'].rolling(window=lma_window).mean()

    # Compare Volatility-Adjusted Price Oscillator with LMA
    conditions = [
        df['vol_adj_price_oscillator'] > df['lma'],
        df['vol_adj_price_oscillator'] < df['lma']
    ]
    choices = [1, -1]
    df['final_alpha_factor'] = pd.Series(pd.np.select(conditions, choices, default=0), index=df.index)

    return df['final_alpha_factor']
