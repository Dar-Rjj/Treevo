import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced momentum alignment with dynamic regime filtering and volume-amount divergence.
    
    Economic intuition:
    - Multi-timeframe momentum alignment captures persistent price trends
    - Volume-amount divergence identifies shifts in market participation patterns
    - Dynamic regime filtering adapts to changing market volatility conditions
    - The combination provides regime-aware signals for future return prediction
    
    Interpretation:
    - High positive values: strong aligned momentum with institutional flow in stable regimes
    - High negative values: strong reversal signals with retail-driven volume in volatile markets
    - Values near zero: conflicting signals or normal market conditions
    """
    
    # Multi-timeframe momentum alignment with weighted convergence
    momentum_3 = df['close'] / df['close'].shift(3) - 1
    momentum_8 = df['close'] / df['close'].shift(8) - 1
    momentum_21 = df['close'] / df['close'].shift(21) - 1
    
    # Directional alignment score (positive when all timeframes agree)
    directional_alignment = np.sign(momentum_3) * np.sign(momentum_8) * np.sign(momentum_21)
    
    # Magnitude-weighted momentum convergence
    momentum_weights = np.array([0.4, 0.35, 0.25])  # Higher weight for shorter timeframes
    momentum_magnitudes = np.column_stack([abs(momentum_3), abs(momentum_8), abs(momentum_21)])
    weighted_momentum = np.sum(momentum_magnitudes * momentum_weights, axis=1)
    
    # Combined momentum alignment factor
    momentum_alignment = directional_alignment * weighted_momentum
    
    # Volume-amount divergence with trend acceleration
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    amount_ma_10 = df['amount'].rolling(window=10).mean()
    
    volume_acceleration = df['volume'] / volume_ma_10 - 1
    amount_acceleration = df['amount'] / amount_ma_10 - 1
    
    # Divergence captures institutional vs retail flow patterns
    volume_amount_divergence = volume_acceleration - amount_acceleration
    
    # Dynamic regime filtering using adaptive volatility bands
    true_range = np.maximum(df['high'] - df['low'],
                           np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    
    # Volatility regime using rolling percentiles
    low_vol_threshold = true_range.rolling(window=20).apply(lambda x: np.percentile(x, 30))
    high_vol_threshold = true_range.rolling(window=20).apply(lambda x: np.percentile(x, 70))
    
    # Regime scoring: positive for low volatility, negative for high volatility
    regime_score = np.where(true_range < low_vol_threshold, 1.0,
                           np.where(true_range > high_vol_threshold, -1.0, 0.0))
    
    # Price efficiency confirmation using intraday range utilization
    close_to_close_move = abs(df['close'] - df['close'].shift(1))
    daily_range = df['high'] - df['low']
    range_efficiency = close_to_close_move / (daily_range + 1e-7)
    efficiency_trend = range_efficiency.rolling(window=5).mean()
    
    # Composite factor with regime-adaptive weighting
    factor = (momentum_alignment * 
             (1 + np.tanh(volume_amount_divergence)) *  # Volume-amount enhancement
             (1 + 0.5 * regime_score) *  # Regime adaptation (boost in low vol, dampen in high vol)
             (1.5 - efficiency_trend))  # Efficiency premium (higher for trending markets)
    
    return pd.Series(factor, index=df.index)
