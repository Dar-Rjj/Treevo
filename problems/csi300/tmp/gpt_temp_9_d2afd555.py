import pandas as pd
import numpy as np

def heuristics_v2(df):
    momentum_acceleration = (df['close'].diff(2) - 2 * df['close'].diff(1)) / (df['high'] - df['low'] + 1e-8)
    liquidity_adjustment = np.log(df['volume'] * df['close']) / np.log(df['volume'] * df['close']).rolling(window=5).std()
    adjusted_acceleration = momentum_acceleration * liquidity_adjustment
    
    reversal = (df['close'] - df['close'].rolling(window=3).mean()) / df['close'].rolling(window=3).std()
    volatility_regime = df['close'].rolling(window=10).std() / df['close'].rolling(window=30).std()
    regime_filtered_reversal = reversal * volatility_regime
    
    price_entropy = -((df['close'] / df['close'].shift(1)).rolling(window=5).apply(lambda x: (x / x.sum() * np.log(x / x.sum())).sum(), raw=False))
    dynamic_weight = 1 / (1 + np.exp(-price_entropy))
    
    heuristics_matrix = adjusted_acceleration * dynamic_weight + regime_filtered_reversal * (1 - dynamic_weight)
    return heuristics_matrix
