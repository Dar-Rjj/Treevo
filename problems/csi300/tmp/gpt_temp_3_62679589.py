import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Adjusted Daily Returns
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['volume_adjusted_return'] = df['daily_return'] * df['volume']

    # Intraday Composite Score
    df['high_low_delta'] = df['high'] - df['low']
    df['close_open_delta'] = df['close'] - df['open']
    df['sum_of_deltas'] = df['high_low_delta'] + df['close_open_delta']
    df['volume_ema'] = df['volume'].ewm(span=20, adjust=False).mean()
    df['intraday_composite_score'] = df['sum_of_deltas'] * df['volume'] / df['volume_ema']

    # Weighted Momentum
    df['ma_volume_adjusted_return'] = df['volume_adjusted_return'].rolling(window=10).mean().shift(1)
    df['momentum_factor'] = df['ma_volume_adjusted_return'] / df['volume_ema']

    # Volatility
    df['daily_volatility'] = df['daily_return'].rolling(window=10).std().shift(1)
    df['inverse_volatility'] = 1 / (df['daily_volatility'] + 1e-8)

    # Divergence
    df['ema_returns'] = df['daily_return'].ewm(span=20, adjust=False).mean()
    df['squared_returns'] = df['daily_return'] ** 2
    df['ema_squared_returns'] = df['squared_returns'].ewm(span=20, adjust=False).mean()
    df['divergence'] = np.abs(df['ema_returns'] - np.sqrt(df['ema_squared_returns']))

    # Intraday Movement Factor
    df['true_range'] = df[['high', 'low', df['close'].shift(1)]].max(axis=1) - df[['high', 'low', df['close'].shift(1)]].min(axis=1)
    df['intraday_high_low_spread'] = df['high'] - df['low']
    df['normalized_intraday_spread'] = df['intraday_high_low_spread'] / df['true_range'].shift(1)

    # Close Price Direction
    df['price_direction'] = df['close'] - df['open']

    # Volume Impact
    df['volume_change'] = df['volume'] - df['volume'].shift(1)

    # Intraday Price Range Ratio
    df['intraday_price_range_ratio'] = (df['high'] - df['low']) / df['low']

    # Volume Adjusted Close
    df['volume_adjusted_close'] = df['close'] * np.sqrt(df['volume'])

    # Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']

    # Final Alpha Factor
    df['alpha_factor'] = (
        df['intraday_composite_score'] + 
        df['momentum_factor'] * df['inverse_volatility'] * 
        df['intraday_price_range_ratio'] * 
        df['volume_adjusted_close'] * 
        df['intraday_return'] * 
        df['volume_change']
    )

    return df['alpha_factor']
