import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum-Volume Divergence Alpha Factor
    """
    df = data.copy()
    
    # Price Momentum Components
    df['price_momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['price_momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # Volume Momentum Components
    df['volume_momentum_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_momentum_10d'] = df['volume'] / df['volume'].shift(10) - 1
    df['volume_momentum_20d'] = df['volume'] / df['volume'].shift(20) - 1
    
    # True Range for Volatility
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volume Regime Detection
    df['volume_acceleration'] = (df['volume'] / df['volume'].shift(5)) - (df['volume'].shift(5) / df['volume'].shift(10))
    
    # Volume regime classification
    df['volume_regime'] = np.where(df['volume_acceleration'] > 0, 'expanding',
                                  np.where(df['volume_acceleration'] < 0, 'contracting', 'stable'))
    
    # Volume regime persistence
    df['volume_regime_change'] = df['volume_regime'] != df['volume_regime'].shift(1)
    df['volume_consecutive_days'] = df.groupby(df['volume_regime_change'].cumsum()).cumcount() + 1
    df['volume_strength'] = np.minimum(df['volume_consecutive_days'] / 5, 1.0)
    
    # Volatility Regime Detection
    df['volatility_acceleration'] = (df['true_range'] / df['true_range'].shift(5)) - (df['true_range'].shift(5) / df['true_range'].shift(10))
    
    # Volatility regime classification
    df['volatility_regime'] = np.where(df['volatility_acceleration'] > 0, 'high',
                                      np.where(df['volatility_acceleration'] < 0, 'low', 'normal'))
    
    # Volatility regime persistence
    df['volatility_regime_change'] = df['volatility_regime'] != df['volatility_regime'].shift(1)
    df['volatility_consecutive_days'] = df.groupby(df['volatility_regime_change'].cumsum()).cumcount() + 1
    df['volatility_strength'] = np.minimum(df['volatility_consecutive_days'] / 5, 1.0)
    
    # Momentum-Volume Divergence
    df['divergence_5d'] = df['price_momentum_5d'] - df['volume_momentum_5d']
    df['divergence_10d'] = df['price_momentum_10d'] - df['volume_momentum_10d']
    df['divergence_20d'] = df['price_momentum_20d'] - df['volume_momentum_20d']
    
    # Divergence Acceleration
    df['divergence_accel_5d'] = df['divergence_5d'] - df['divergence_5d'].shift(5)
    df['divergence_accel_10d'] = df['divergence_10d'] - df['divergence_10d'].shift(5)
    df['divergence_accel_20d'] = df['divergence_20d'] - df['divergence_20d'].shift(5)
    
    # Cross-sectional ranking within each date
    def rank_within_date(series):
        return series.rank(pct=True)
    
    # Divergence ranks
    df['div_rank_5d'] = df.groupby(df.index)['divergence_5d'].transform(rank_within_date)
    df['div_rank_10d'] = df.groupby(df.index)['divergence_10d'].transform(rank_within_date)
    df['div_rank_20d'] = df.groupby(df.index)['divergence_20d'].transform(rank_within_date)
    
    # Acceleration ranks
    df['accel_rank_5d'] = df.groupby(df.index)['divergence_accel_5d'].transform(rank_within_date)
    df['accel_rank_10d'] = df.groupby(df.index)['divergence_accel_10d'].transform(rank_within_date)
    df['accel_rank_20d'] = df.groupby(df.index)['divergence_accel_20d'].transform(rank_within_date)
    
    # Regime-Adaptive Weighting
    def get_volume_weights(regime):
        if regime == 'expanding':
            return [0.5, 0.3, 0.2]  # Emphasize short-term
        elif regime == 'contracting':
            return [0.3, 0.4, 0.3]  # Balanced with slight emphasis on acceleration
        else:  # stable
            return [0.33, 0.33, 0.34]  # Balanced
    
    def get_volatility_weights(regime):
        if regime == 'high':
            return [0.4, 0.4, 0.2]  # Higher weight to volume confirmation
        elif regime == 'low':
            return [0.2, 0.4, 0.4]  # Higher weight to momentum persistence
        else:  # normal
            return [0.33, 0.33, 0.34]  # Equal weighting
    
    # Apply regime weights
    volume_weights = df['volume_regime'].apply(get_volume_weights)
    volatility_weights = df['volatility_regime'].apply(get_volatility_weights)
    
    df['volume_weight_5d'] = volume_weights.apply(lambda x: x[0])
    df['volume_weight_10d'] = volume_weights.apply(lambda x: x[1])
    df['volume_weight_20d'] = volume_weights.apply(lambda x: x[2])
    
    df['volatility_weight_5d'] = volatility_weights.apply(lambda x: x[0])
    df['volatility_weight_10d'] = volatility_weights.apply(lambda x: x[1])
    df['volatility_weight_20d'] = volatility_weights.apply(lambda x: x[2])
    
    # Composite factor construction
    # Base divergence score with regime-adaptive weights
    df['weighted_divergence'] = (
        df['div_rank_5d'] * df['volume_weight_5d'] * df['volatility_weight_5d'] +
        df['div_rank_10d'] * df['volume_weight_10d'] * df['volatility_weight_10d'] +
        df['div_rank_20d'] * df['volume_weight_20d'] * df['volatility_weight_20d']
    )
    
    # Acceleration adjustment
    df['acceleration_adjustment'] = (
        df['accel_rank_5d'] * df['volume_weight_5d'] +
        df['accel_rank_10d'] * df['volume_weight_10d'] +
        df['accel_rank_20d'] * df['volume_weight_20d']
    )
    
    # Regime confidence multiplier
    df['regime_confidence'] = df['volume_strength'] * df['volatility_strength']
    
    # Final alpha factor
    df['alpha_factor'] = (
        df['weighted_divergence'] * 
        (1 + 0.5 * df['acceleration_adjustment']) * 
        df['regime_confidence']
    )
    
    # Normalize to cross-sectional z-score
    def zscore_normalize(series):
        return (series - series.mean()) / series.std()
    
    df['final_alpha'] = df.groupby(df.index)['alpha_factor'].transform(zscore_normalize)
    
    return df['final_alpha']
