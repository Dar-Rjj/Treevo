import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum acceleration with volume efficiency and volatility normalization
    # Uses non-linear transforms and relative comparisons for better signal quality
    
    # Price momentum acceleration (2nd derivative of price)
    short_momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    long_momentum = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    momentum_acceleration = short_momentum - long_momentum
    
    # Volume efficiency: volume relative to price range efficiency
    price_range_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    volume_efficiency = df['volume'] * np.abs(price_range_efficiency)
    volume_efficiency_rel = volume_efficiency / volume_efficiency.rolling(window=15).mean()
    
    # Volatility normalization using rolling percentiles
    true_range = np.maximum(
        np.maximum(
            df['high'] - df['low'],
            np.abs(df['high'] - df['close'].shift(1))
        ),
        np.abs(df['low'] - df['close'].shift(1))
    )
    vol_normalized = true_range / true_range.rolling(window=20).apply(
        lambda x: np.percentile(x.dropna(), 70) if len(x.dropna()) > 0 else 1.0
    )
    
    # Non-linear combination with economic rationale:
    # Strong momentum acceleration amplified by efficient volume, 
    # normalized by relative volatility to identify quality moves
    factor = (momentum_acceleration * np.tanh(volume_efficiency_rel * 2)) / (vol_normalized + 1e-7)
    
    return factor
