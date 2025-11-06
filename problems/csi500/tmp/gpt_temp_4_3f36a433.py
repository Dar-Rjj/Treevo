import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum acceleration with volume divergence and volatility scaling
    # Uses acceleration (change in momentum) combined with volume divergence from normal patterns
    
    # Price momentum acceleration (5-day vs 10-day momentum difference)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_10d = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    momentum_acceleration = momentum_5d - momentum_10d
    
    # Volume divergence: current volume vs expected volume based on price movement
    price_range_ratio = (df['high'] - df['low']) / df['close']
    expected_volume = df['volume'].rolling(window=20).mean() * (1 + price_range_ratio * 2)
    volume_divergence = df['volume'] / (expected_volume + 1e-7)
    
    # Non-linear transform of momentum acceleration (sigmoid-like function)
    transformed_momentum = momentum_acceleration / (1 + abs(momentum_acceleration))
    
    # Relative volatility scaling using rolling percentile
    volatility_20d = (df['high'] - df['low']).rolling(window=20).std()
    vol_percentile = volatility_20d.rolling(window=60).apply(lambda x: (x.iloc[-1] > x).mean())
    
    # Volume confirmation strength (absolute vs relative volume)
    volume_strength = (df['volume'] / df['volume'].rolling(window=20).mean()) * volume_divergence
    
    # Final factor: accelerated momentum amplified by volume divergence, scaled by relative volatility
    factor = transformed_momentum * volume_strength * (1 + vol_percentile)
    
    return factor
