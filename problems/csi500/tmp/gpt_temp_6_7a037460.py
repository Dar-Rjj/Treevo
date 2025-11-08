import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Volume Divergence Alpha Factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic momentum components
    df['price_momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['price_momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    
    df['volume_momentum_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_momentum_10d'] = df['volume'] / df['volume'].shift(10) - 1
    df['volume_momentum_20d'] = df['volume'] / df['volume'].shift(20) - 1
    
    # Exponential smoothing of momentum components
    alpha = 0.3
    df['ema_price_5d'] = df['price_momentum_5d'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_price_10d'] = df['price_momentum_10d'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_price_20d'] = df['price_momentum_20d'].ewm(alpha=alpha, adjust=False).mean()
    
    df['ema_volume_5d'] = df['volume_momentum_5d'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_volume_10d'] = df['volume_momentum_10d'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_volume_20d'] = df['volume_momentum_20d'].ewm(alpha=alpha, adjust=False).mean()
    
    # Momentum acceleration
    df['price_accel_5d'] = df['ema_price_5d'] - df['ema_price_5d'].shift(1)
    df['price_accel_10d'] = df['ema_price_10d'] - df['ema_price_10d'].shift(1)
    df['price_accel_20d'] = df['ema_price_20d'] - df['ema_price_20d'].shift(1)
    
    df['volume_accel_5d'] = df['ema_volume_5d'] - df['ema_volume_5d'].shift(1)
    df['volume_accel_10d'] = df['ema_volume_10d'] - df['ema_volume_10d'].shift(1)
    df['volume_accel_20d'] = df['ema_volume_20d'] - df['ema_volume_20d'].shift(1)
    
    # Amount-based regime detection
    df['amount_momentum_5d'] = df['amount'] / df['amount'].shift(5) - 1
    df['amount_momentum_10d'] = df['amount'] / df['amount'].shift(10) - 1
    df['amount_momentum_20d'] = df['amount'] / df['amount'].shift(20) - 1
    
    # Regime classification
    high_participation = (df['amount_momentum_5d'] > 0.1) & (df['amount_momentum_10d'] > 0.05)
    low_participation = (df['amount_momentum_5d'] < -0.1) & (df['amount_momentum_10d'] < -0.05)
    
    # Regime persistence
    regime_persistence = pd.Series(index=df.index, dtype=int)
    current_regime = None
    persistence_count = 0
    
    for i in range(len(df)):
        if high_participation.iloc[i]:
            regime = 'high'
        elif low_participation.iloc[i]:
            regime = 'low'
        else:
            regime = 'normal'
        
        if regime == current_regime:
            persistence_count += 1
        else:
            persistence_count = 1
            current_regime = regime
        
        regime_persistence.iloc[i] = persistence_count
    
    df['regime_persistence'] = regime_persistence
    df['amount_acceleration'] = df['amount_momentum_5d'] - df['amount_momentum_10d']
    
    # Volatility assessment
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['tr_20d_avg'] = df['tr'].rolling(window=20).mean()
    
    high_volatility = df['tr'] > (1.5 * df['tr_20d_avg'])
    low_volatility = df['tr'] < (0.7 * df['tr_20d_avg'])
    
    # Divergence calculation
    df['raw_divergence_5d'] = df['ema_price_5d'] - df['ema_volume_5d']
    df['raw_divergence_10d'] = df['ema_price_10d'] - df['ema_volume_10d']
    df['raw_divergence_20d'] = df['ema_price_20d'] - df['ema_volume_20d']
    
    df['accel_divergence_5d'] = df['price_accel_5d'] - df['volume_accel_5d']
    df['accel_divergence_10d'] = df['price_accel_10d'] - df['volume_accel_10d']
    df['accel_divergence_20d'] = df['price_accel_20d'] - df['volume_accel_20d']
    
    # Cross-sectional ranking (within each day)
    for date in df.index:
        day_data = df.loc[date]
        
        if not isinstance(day_data, pd.Series):  # Skip if we have multiple stocks
            continue
            
        # For single stock implementation, use absolute values for ranking simulation
        rank_div_5d = 0.5  # Placeholder for cross-sectional rank
        rank_div_10d = 0.5
        rank_div_20d = 0.5
        rank_accel_5d = 0.5
        rank_accel_10d = 0.5
        rank_accel_20d = 0.5
        
        # Regime-adaptive weighting
        if high_participation.loc[date]:
            price_weight = 0.3
            volume_weight = 0.7
            timeframe_weights = {'5d': 0.2, '10d': 0.5, '20d': 0.3}
        elif low_participation.loc[date]:
            price_weight = 0.7
            volume_weight = 0.3
            timeframe_weights = {'5d': 0.2, '10d': 0.3, '20d': 0.5}
        else:
            price_weight = 0.5
            volume_weight = 0.5
            timeframe_weights = {'5d': 0.33, '10d': 0.33, '20d': 0.34}
        
        # Volatility regime adjustment
        if high_volatility.loc[date]:
            timeframe_weights = {'5d': 0.6, '10d': 0.3, '20d': 0.1}
        elif low_volatility.loc[date]:
            timeframe_weights = {'5d': 0.2, '10d': 0.3, '20d': 0.5}
        
        # Core divergence score
        divergence_score = (
            rank_div_5d * timeframe_weights['5d'] +
            rank_div_10d * timeframe_weights['10d'] +
            rank_div_20d * timeframe_weights['20d']
        )
        
        acceleration_score = (
            rank_accel_5d * timeframe_weights['5d'] +
            rank_accel_10d * timeframe_weights['10d'] +
            rank_accel_20d * timeframe_weights['20d']
        )
        
        # Regime confidence multiplier
        regime_strength = min(df['regime_persistence'].loc[date] / 10, 1.0)
        amount_strength = min(abs(df['amount_acceleration'].loc[date]) * 10, 1.0)
        regime_confidence = 0.6 * regime_strength + 0.4 * amount_strength
        
        # Final alpha construction
        final_divergence = (
            price_weight * df['ema_price_10d'].loc[date] -
            volume_weight * df['ema_volume_10d'].loc[date]
        )
        
        acceleration_enhancement = acceleration_score * 0.3
        
        alpha_value = (final_divergence + acceleration_enhancement) * regime_confidence
        
        result.loc[date] = alpha_value
    
    # Handle NaN values
    result = result.fillna(0)
    
    return result
