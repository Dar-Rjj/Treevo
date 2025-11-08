import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Momentum-Volume Divergence Alpha Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Components
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['price_momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume Momentum Components
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_momentum_20d'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Exponential Smoothing Framework
    alpha = 0.3
    for col in ['price_momentum_5d', 'price_momentum_10d', 'price_momentum_20d',
                'volume_momentum_5d', 'volume_momentum_10d', 'volume_momentum_20d']:
        data[f'ema_{col}'] = data[col].ewm(alpha=alpha, adjust=False).mean()
    
    # Momentum Acceleration Calculation
    data['price_momentum_accel'] = data['ema_price_momentum_20d'] - data['ema_price_momentum_20d'].shift(1)
    data['volume_momentum_accel'] = data['ema_volume_momentum_20d'] - data['ema_volume_momentum_20d'].shift(1)
    
    # Regime Detection System
    # Amount-based regime
    data['amount_20d_ma'] = data['amount'].rolling(window=20).mean()
    data['amount_acceleration'] = data['amount'] / data['amount'].shift(5) - 1
    
    # Volatility assessment
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['volatility_20d'] = data['daily_range'].rolling(window=20).std()
    
    # Classify regimes
    data['amount_regime'] = np.where(data['amount_acceleration'] > data['amount_acceleration'].rolling(20).quantile(0.7), 
                                    'high_participation', 'low_participation')
    data['vol_regime'] = np.where(data['volatility_20d'] > data['volatility_20d'].rolling(20).quantile(0.7),
                                 'high_vol', 'low_vol')
    
    # Momentum-Volume Divergence Core
    data['momentum_divergence'] = (data['ema_price_momentum_20d'] - data['ema_volume_momentum_20d']) * \
                                 (data['price_momentum_accel'] - data['volume_momentum_accel'])
    
    # Dynamic Cross-Sectional Ranking
    # Calculate cross-sectional z-scores for momentum divergence
    data['divergence_zscore'] = data.groupby(data.index)['momentum_divergence'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    # Volatility-normalized ranking
    data['volatility_rank'] = data.groupby(data.index)['volatility_20d'].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') if len(x.unique()) >= 5 else 0
    )
    
    # Rank within volatility buckets
    data['vol_bucket_rank'] = data.groupby(['volatility_rank', data.index])['momentum_divergence'].transform(
        lambda x: x.rank(pct=True) if len(x) > 1 else 0.5
    )
    
    # Regime-Adaptive Factor Construction
    def calculate_regime_weight(row):
        if row['vol_regime'] == 'high_vol':
            # High volatility: emphasize volume confirmation
            return 0.7  # 70% volume, 30% price
        elif row['vol_regime'] == 'low_vol':
            # Low volatility: emphasize price persistence
            return 0.3  # 30% volume, 70% price
        else:
            # Transition regime
            return 0.5  # Equal weighting
    
    data['regime_weight'] = data.apply(calculate_regime_weight, axis=1)
    
    # Apply regime-based weighting to divergence signal
    data['regime_adaptive_divergence'] = (
        data['regime_weight'] * data['volume_momentum_accel'] + 
        (1 - data['regime_weight']) * data['price_momentum_accel']
    ) * data['momentum_divergence']
    
    # Final Alpha Output - Composite score
    data['alpha_factor'] = (
        data['regime_adaptive_divergence'] * 
        data['vol_bucket_rank'] * 
        np.tanh(data['divergence_zscore'])  # Normalize extreme values
    )
    
    # Clean up and return
    result = data['alpha_factor'].copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    
    return result
