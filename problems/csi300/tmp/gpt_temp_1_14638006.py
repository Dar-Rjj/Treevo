import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate novel alpha factor combining adaptive momentum, volume confirmation, 
    and volatility normalization for stock return prediction.
    """
    df = data.copy()
    
    # Adaptive Momentum Component
    # Multi-timeframe Momentum Signals
    df['ultra_short_mom'] = df['close'] / df['close'].shift(1) - 1
    df['short_term_mom'] = df['close'] / df['close'].shift(3) - 1
    df['medium_term_mom'] = df['close'] / df['close'].shift(5) - 1
    df['mom_acceleration'] = (df['close'] / df['close'].shift(1)) - (df['close'] / df['close'].shift(3))
    
    # Volatility-Adaptive Weighting
    df['current_vol'] = (df['high'] - df['low']) / df['close']
    df['hist_vol_baseline'] = df['current_vol'].rolling(window=10, min_periods=5).mean()
    df['vol_regime'] = df['current_vol'] / df['hist_vol_baseline']
    
    # Dynamic momentum weighting based on volatility regime
    def adaptive_momentum(row):
        if pd.isna(row['vol_regime']):
            return np.nan
        
        if row['vol_regime'] > 1.2:  # High volatility regime
            weights = [0.2, 0.5, 0.2, 0.1]  # Emphasize short-term
        elif row['vol_regime'] < 0.8:  # Low volatility regime
            weights = [0.1, 0.3, 0.5, 0.1]  # Emphasize medium-term
        else:  # Normal volatility
            weights = [0.25, 0.35, 0.25, 0.15]  # Balanced weighting
        
        mom_signals = [row['ultra_short_mom'], row['short_term_mom'], 
                      row['medium_term_mom'], row['mom_acceleration']]
        
        if any(pd.isna(signal) for signal in mom_signals):
            return np.nan
        
        return sum(w * s for w, s in zip(weights, mom_signals))
    
    df['adaptive_momentum'] = df.apply(adaptive_momentum, axis=1)
    
    # Volume Confirmation Engine
    # Volume Intensity Analysis
    df['volume_ma_5'] = df['volume'].rolling(window=5, min_periods=3).mean()
    df['volume_surge'] = df['volume'] / df['volume_ma_5'].shift(1)
    df['volume_trend'] = df['volume'] / df['volume'].shift(3) - 1
    
    # Volume persistence (count of consecutive volume increases)
    df['volume_persistence'] = (
        (df['volume'] > df['volume'].shift(1)).astype(int) +
        (df['volume'] > df['volume'].shift(2)).astype(int) +
        (df['volume'] > df['volume'].shift(3)).astype(int)
    )
    
    # Price-Volume Alignment
    def volume_confirmation_score(row):
        if any(pd.isna([row['adaptive_momentum'], row['volume_surge'], 
                       row['volume_trend'], row['volume_persistence']])):
            return np.nan
        
        # Base score from volume intensity
        base_score = 0.0
        
        # Volume surge contribution
        if row['volume_surge'] > 1.5:
            base_score += 0.4
        elif row['volume_surge'] > 1.2:
            base_score += 0.2
        elif row['volume_surge'] < 0.8:
            base_score -= 0.2
        
        # Volume trend contribution
        if row['volume_trend'] > 0.1:
            base_score += 0.3
        elif row['volume_trend'] < -0.1:
            base_score -= 0.3
        
        # Persistence bonus
        persistence_bonus = row['volume_persistence'] * 0.1
        base_score += persistence_bonus
        
        # Price-volume alignment multiplier
        momentum_direction = 1 if row['adaptive_momentum'] > 0 else -1
        volume_direction = 1 if row['volume_trend'] > 0 else -1
        
        if momentum_direction == volume_direction:
            alignment_multiplier = 1.5  # Strong confirmation
        elif momentum_direction * volume_direction < 0:
            alignment_multiplier = 0.5  # Divergence - reduce confidence
        else:
            alignment_multiplier = 1.0  # Neutral
        
        final_score = base_score * alignment_multiplier
        return np.clip(final_score, -1, 1)  # Bound between -1 and 1
    
    df['volume_confirmation'] = df.apply(volume_confirmation_score, axis=1)
    
    # Volatility-Normalized Composite
    # Dynamic Volatility Scaling
    df['range_vol'] = (df['high'] - df['low']) / df['close']
    df['close_vol'] = (df['close'] / df['close'].shift(1)).rolling(window=10, min_periods=5).std()
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['true_range_vol'] = df['true_range'] / df['close']
    
    # Combined volatility measure
    df['current_volatility'] = (
        0.4 * df['range_vol'] + 
        0.3 * df['close_vol'] + 
        0.3 * df['true_range_vol']
    )
    
    # Raw composite
    df['raw_composite'] = df['adaptive_momentum'] * df['volume_confirmation']
    
    # Volatility normalization with regime adjustment
    def volatility_scaling(row):
        if any(pd.isna([row['raw_composite'], row['current_volatility'], row['vol_regime']])):
            return np.nan
        
        base_scaling = row['raw_composite'] / (row['current_volatility'] + 1e-8)
        
        # Regime adjustment
        if row['vol_regime'] > 1.5:
            regime_adj = 0.7  # Reduce signal in extreme volatility
        elif row['vol_regime'] > 1.2:
            regime_adj = 0.9
        elif row['vol_regime'] < 0.5:
            regime_adj = 0.8  # Reduce signal in extremely low volatility
        else:
            regime_adj = 1.0
        
        return base_scaling * regime_adj
    
    df['vol_scaled_factor'] = df.apply(volatility_scaling, axis=1)
    
    # Cross-Framework Validation
    def cross_validation_score(row):
        if any(pd.isna([row['adaptive_momentum'], row['volume_confirmation'], 
                       row['current_volatility'], row['vol_scaled_factor']])):
            return np.nan
        
        consistency_score = 0
        
        # Momentum-volume-volatility consistency
        momentum_strength = abs(row['adaptive_momentum'])
        volume_strength = abs(row['volume_confirmation'])
        volatility_level = row['current_volatility']
        
        # Strongest signal: high momentum + high volume + low volatility
        if (momentum_strength > 0.02 and volume_strength > 0.5 and volatility_level < 0.03):
            consistency_score += 1.0
        # Weakest signal: low momentum + low volume + high volatility
        elif (momentum_strength < 0.005 and volume_strength < 0.2 and volatility_level > 0.05):
            consistency_score -= 0.5
        # Divergent signals
        elif (row['adaptive_momentum'] * row['volume_confirmation'] < 0):
            consistency_score -= 0.3
        
        # Signal quality based on historical consistency (simplified)
        if momentum_strength > 0.01 and volume_strength > 0.3:
            consistency_score += 0.2
        
        return np.clip(consistency_score, -1, 1)
    
    df['cross_validation'] = df.apply(cross_validation_score, axis=1)
    
    # Final Alpha Output
    def final_alpha_score(row):
        if any(pd.isna([row['vol_scaled_factor'], row['cross_validation']])):
            return np.nan
        
        # Combine volatility-scaled factor with cross-validation
        base_signal = row['vol_scaled_factor']
        validation_boost = 1 + (row['cross_validation'] * 0.3)  # ±30% adjustment
        
        final_signal = base_signal * validation_boost
        
        return np.clip(final_signal, -2, 2)  # Reasonable bounds
    
    df['alpha_factor'] = df.apply(final_alpha_score, axis=1)
    
    return df['alpha_factor']
