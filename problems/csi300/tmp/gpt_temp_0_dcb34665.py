import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Parameters
    n = 5  # Number of days for volume-weighted momentum
    m = 10  # Number of days for adaptive volatility
    k = 7  # Number of days for EMA
    p = 5  # Number of days for liquidity adjustment
    smoothing_factor = 0.2  # Smoothing factor for EMA
    sector_ema_weight = 0.3
    liquidity_adjustment_weight = 0.2
    momentum_weight = 0.4
    volatility_weight = 0.1

    # Volume-Weighted Momentum Calculation
    df['close_vol'] = df['close'] * df['volume']
    df['sum_close_vol'] = df['close_vol'].rolling(window=n).sum()
    df['sum_volume'] = df['volume'].rolling(window=n).sum()
    df['volume_weighted_close'] = df['sum_close_vol'] / df['sum_volume']
    df['volume_weighted_momentum'] = df['volume_weighted_close'] - df['volume_weighted_close'].shift(n)

    # Adaptive Volatility Calculation
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['adaptive_volatility'] = df['log_returns'].rolling(window=m).std()

    # Adjust the volume-weighted momentum by the adaptive volatility
    df['adjusted_momentum'] = df['volume_weighted_momentum'] / df['adaptive_volatility']

    # Sector-Specific EMA
    df['sector_ema'] = df.groupby('sector')['close'].transform(lambda x: x.ewm(span=k, adjust=False).mean())

    # Liquidity Adjustment
    df['average_volume'] = df['volume'].rolling(window=p).mean()
    df['liquidity_adjusted_momentum'] = df['volume_weighted_momentum'] / df['average_volume']

    # Combine the weighted factors into a single alpha factor
    df['alpha_factor'] = (momentum_weight * df['adjusted_momentum'] +
                          volatility_weight * df['adaptive_volatility'] +
                          sector_ema_weight * df['sector_ema'] +
                          liquidity_adjustment_weight * df['liquidity_adjusted_momentum'])

    return df['alpha_factor']
