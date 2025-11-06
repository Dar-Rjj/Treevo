import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-scaled momentum with shorter window (3-day) for volatility, longer (10-day) for trend
    momentum_10d = (df['close'] - df['close'].shift(10)) / (df['close'].shift(10) + 1e-7)
    volatility_3d = (df['high'] - df['low']).rolling(window=3).std()
    vol_scaled_momentum = momentum_10d / (volatility_3d + 1e-7)
    
    # Volume-price signals with amount confirmation using multiplicative combination
    volume_change = np.log1p(df['volume'] / (df['volume'].shift(5) + 1e-7))
    amount_change = np.log1p(df['amount'] / (df['amount'].shift(5) + 1e-7))
    price_change = np.log1p(df['close'] / (df['close'].shift(5) + 1e-7))
    volume_price_signal = np.tanh(volume_change * amount_change - price_change)
    
    # Range efficiency with regime adjustment using rolling quantiles
    current_range = df['high'] - df['low']
    avg_range_10d = (df['high'] - df['low']).rolling(window=10).mean()
    range_ratio = current_range / (avg_range_10d + 1e-7)
    range_regime = range_ratio.rolling(window=20).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    range_efficiency = np.tanh(np.log1p(range_ratio / (range_regime + 1e-7)))
    
    # Multiplicative combination with bounded outputs
    alpha_factor = (
        np.tanh(vol_scaled_momentum) * 
        (1 + volume_price_signal) * 
        (1 + range_efficiency)
    )
    
    return alpha_factor
