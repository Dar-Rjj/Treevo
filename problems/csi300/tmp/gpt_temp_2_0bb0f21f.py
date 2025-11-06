import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-normalized momentum, volume-price divergence,
    and volatility regime classification.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Volatility-Normalized Momentum Framework
    # Multi-timeframe momentum calculation
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_10d = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    momentum_20d = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Volatility scaling using price range
    current_vol = (df['high'] - df['low']) / df['close']
    rolling_vol_5d = ((df['high'] - df['low']) / df['close']).rolling(window=5).mean()
    
    # Volatility-adjusted momentum
    vol_adj_momentum_3d = momentum_3d / rolling_vol_5d
    vol_adj_momentum_5d = momentum_5d / rolling_vol_5d
    vol_adj_momentum_10d = momentum_10d / rolling_vol_5d
    vol_adj_momentum_20d = momentum_20d / rolling_vol_5d
    
    # Momentum convergence analysis
    vol_adj_momentums = pd.DataFrame({
        'm3': vol_adj_momentum_3d,
        'm5': vol_adj_momentum_5d,
        'm10': vol_adj_momentum_10d,
        'm20': vol_adj_momentum_20d
    })
    
    direction_consistency = (vol_adj_momentums > 0).sum(axis=1) / 4.0
    
    def calculate_magnitude_dispersion(row):
        valid_values = row.dropna()
        if len(valid_values) == 0:
            return 1.0
        if abs(valid_values.mean()) < 1e-10:
            return 1.0
        return (valid_values.max() - valid_values.min()) / abs(valid_values.mean())
    
    magnitude_dispersion = vol_adj_momentums.apply(calculate_magnitude_dispersion, axis=1)
    convergence_score = direction_consistency * (1 - magnitude_dispersion.clip(upper=1.0))
    
    # Core momentum factor
    core_momentum = vol_adj_momentum_5d * convergence_score
    
    # Volume-Price Divergence Framework
    # Volume trend analysis
    avg_volume_4d = df['volume'].shift(1).rolling(window=4).mean()
    volume_momentum = (df['volume'] - avg_volume_4d) / avg_volume_4d
    
    volume_acceleration = (df['volume'] / df['volume'].shift(2)) - (df['volume'].shift(2) / df['volume'].shift(4))
    
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_above_avg = df['volume'] > volume_5d_avg
    
    def calculate_persistence(series):
        persistence = pd.Series(index=series.index, dtype=float)
        current_count = 0
        for i in range(len(series)):
            if series.iloc[i]:
                current_count += 1
            else:
                current_count = 0
            persistence.iloc[i] = current_count
        return persistence
    
    persistence_count = calculate_persistence(volume_above_avg)
    
    # Divergence detection
    price_momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    bullish_divergence = (price_momentum_5d < 0) & (volume_momentum > 0)
    bearish_divergence = (price_momentum_5d > 0) & (volume_momentum < 0)
    confirmation = (price_momentum_5d > 0) & (volume_momentum > 0) | (price_momentum_5d < 0) & (volume_momentum < 0)
    
    # Divergence strength scoring
    base_divergence_score = pd.Series(0.0, index=df.index)
    base_divergence_score[bullish_divergence] = 1.0
    base_divergence_score[bearish_divergence] = -1.0
    base_divergence_score[confirmation] = 0.0
    
    strength_multiplier = 1 + abs(volume_acceleration)
    persistence_enhancement = 1 + (persistence_count * 0.05)
    
    divergence_score = base_divergence_score * strength_multiplier * persistence_enhancement
    
    # Volume divergence adjustment
    volume_adjusted = core_momentum * (1 + divergence_score)
    
    # Volatility Regime Classification
    vol_ratio = rolling_vol_5d / ((df['high'] - df['low']) / df['close']).rolling(window=20).mean()
    
    # Initialize regime with hysteresis
    regime = pd.Series('normal', index=df.index)
    prev_regime = 'normal'
    
    for i in range(len(df)):
        if i < 20:
            regime.iloc[i] = 'normal'
            continue
            
        current_ratio = vol_ratio.iloc[i]
        
        if (current_ratio > 1.2) or (current_ratio > 1.1 and prev_regime == 'high'):
            regime.iloc[i] = 'high'
        elif (current_ratio < 0.8) or (current_ratio < 0.9 and prev_regime == 'low'):
            regime.iloc[i] = 'low'
        else:
            regime.iloc[i] = 'normal'
        
        prev_regime = regime.iloc[i]
    
    # Regime-specific coefficients
    regime_coefficient = pd.Series(1.0, index=df.index)
    regime_coefficient[regime == 'high'] = 0.7
    regime_coefficient[regime == 'low'] = 1.3
    
    # Final alpha factor
    alpha_factor = volume_adjusted * regime_coefficient
    
    # Fill NaN values with 0
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
