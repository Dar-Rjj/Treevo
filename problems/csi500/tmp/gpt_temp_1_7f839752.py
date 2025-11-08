import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Detection
    std_10 = df['close'].rolling(window=10, min_periods=10).std()
    std_30 = df['close'].rolling(window=30, min_periods=30).std()
    volatility_ratio = std_10 / std_30
    volatility_breakout = (volatility_ratio > 1.2).astype(float)
    
    # Volume-Price Efficiency
    volume_median_10 = df['volume'].rolling(window=10, min_periods=10).median()
    volume_ratio = df['volume'] / volume_median_10
    volume_concentration = volume_ratio.rolling(window=10, min_periods=10).apply(
        lambda x: np.max(x) / np.mean(x) if np.mean(x) != 0 else 0
    )
    
    range_efficiency = (df['high'] - df['low']) / (abs(df['close'] - df['open']) + 1e-8)
    
    # Momentum Divergence
    price_momentum = df['close'] / df['close'].shift(5) - 1
    volume_momentum = df['volume'] / df['volume'].shift(5) - 1
    momentum_divergence = abs(price_momentum - volume_momentum)
    
    # Base Factor
    base_factor = volatility_ratio * volume_concentration * momentum_divergence * range_efficiency
    
    # Conditional Enhancement
    close_ratio_20 = df['close'] / df['close'].shift(20)
    close_ratio_5 = df['close'] / df['close'].shift(5)
    momentum_diff = close_ratio_20 - close_ratio_5
    
    enhanced_factor = base_factor * (1 + volume_momentum) * momentum_diff * 10
    
    # Final Factor
    factor = np.where(
        (volatility_breakout > 0) & (volume_concentration > 1.1),
        enhanced_factor,
        0
    )
    
    return pd.Series(factor, index=df.index)
