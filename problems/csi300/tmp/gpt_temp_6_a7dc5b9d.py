import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum Component
    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))
    df['price_momentum'] = df['log_return'].rolling(window=20).sum()

    # Calculate Volume Surge Component
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_surge'] = df['volume_change'].ewm(span=10, adjust=False).mean()

    # Incorporate Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['true_range'] = df[['high' - 'low', abs('high' - df['close'].shift(1)), abs('low' - df['close'].shift(1))]].max(axis=1)
    df['average_true_range'] = df['true_range'].rolling(window=14).mean()

    # Incorporate Moving Averages
    df['short_term_ma'] = df['close'].rolling(window=5).mean()
    df['long_term_ma'] = df['close'].rolling(window=50).mean()
    df['ma_difference'] = df['short_term_ma'] - df['long_term_ma']

    # Incorporate Relative Strength
    up_days = (df['close'] > df['close'].shift(1)).astype(int)
    down_days = (df['close'] < df['close'].shift(1)).astype(int)
    rs_up = df['close'].pct_change().apply(lambda x: x if x > 0 else 0).rolling(window=14).mean()
    rs_down = df['close'].pct_change().apply(lambda x: -x if x < 0 else 0).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + (rs_up / rs_down)))

    # Standardize Components
    def standardize(series):
        return (series - series.mean()) / series.std()

    df['price_momentum_std'] = standardize(df['price_momentum'])
    df['volume_surge_std'] = standardize(df['volume_surge'])
    df['volatility_std'] = standardize(df['average_true_range'])
    df['ma_difference_std'] = standardize(df['ma_difference'])
    df['rsi_std'] = standardize(df['rsi'])

    # Weighted Sum
    weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Weights for price_momentum, volume_surge, volatility, ma_difference, rsi
    df['alpha_factor'] = (
        weights[0] * df['price_momentum_std'] +
        weights[1] * df['volume_surge_std'] +
        weights[2] * df['volatility_std'] +
        weights[3] * df['ma_difference_std'] +
        weights[4] * df['rsi_std']
    )

    return df['alpha_factor']
