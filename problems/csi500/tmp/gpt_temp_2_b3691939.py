import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Multi-Timeframe Divergence Alpha Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Raw Momentum Calculation
    # Price Momentum
    data['price_mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['price_mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_mom_10d'] = data['close'] / data['close'].shift(10) - 1
    data['price_mom_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume Momentum
    data['volume_mom_3d'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_mom_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_mom_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_mom_20d'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Regime Classification
    # Volume Regime
    data['volume_ratio'] = data['volume'] / ((data['volume'].shift(5) + data['volume'].shift(10)) / 2)
    data['volume_regime'] = np.select(
        [
            data['volume_ratio'] > 1.2,
            data['volume_ratio'] < 0.8,
            (data['volume_ratio'] >= 0.8) & (data['volume_ratio'] <= 1.2)
        ],
        ['high', 'low', 'normal'],
        default='normal'
    )
    
    # Volatility Regime
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_ratio'] = data['daily_range'] / ((data['daily_range'].shift(5) + data['daily_range'].shift(10)) / 2)
    data['volatility_regime'] = np.select(
        [
            data['range_ratio'] > 1.3,
            data['range_ratio'] < 0.7,
            (data['range_ratio'] >= 0.7) & (data['range_ratio'] <= 1.3)
        ],
        ['high', 'low', 'normal'],
        default='normal'
    )
    
    # Divergence Calculation
    data['div_3d'] = data['price_mom_3d'] - data['volume_mom_3d']
    data['div_5d'] = data['price_mom_5d'] - data['volume_mom_5d']
    data['div_10d'] = data['price_mom_10d'] - data['volume_mom_10d']
    data['div_20d'] = data['price_mom_20d'] - data['volume_mom_20d']
    
    # Divergence Quality
    divergence_cols = ['div_3d', 'div_5d', 'div_10d', 'div_20d']
    data['sign_consistency'] = data[divergence_cols].apply(
        lambda x: (x > 0).sum() if (x > 0).sum() >= (x < 0).sum() else (x < 0).sum(), 
        axis=1
    )
    data['direction_strength'] = data['sign_consistency'] / len(divergence_cols)
    
    # Regime-Adaptive Weighting
    # Volume regime weights
    volume_weights = {
        'high': [0.4, 0.3, 0.2, 0.1],
        'normal': [0.25, 0.25, 0.25, 0.25],
        'low': [0.1, 0.2, 0.3, 0.4]
    }
    
    # Volatility regime weights
    volatility_weights = {
        'high': [0.35, 0.3, 0.2, 0.15],
        'normal': [0.25, 0.25, 0.25, 0.25],
        'low': [0.15, 0.2, 0.3, 0.35]
    }
    
    # Apply regime weights
    data['volume_weight_3d'] = data['volume_regime'].map(lambda x: volume_weights[x][0])
    data['volume_weight_5d'] = data['volume_regime'].map(lambda x: volume_weights[x][1])
    data['volume_weight_10d'] = data['volume_regime'].map(lambda x: volume_weights[x][2])
    data['volume_weight_20d'] = data['volume_regime'].map(lambda x: volume_weights[x][3])
    
    data['volatility_weight_3d'] = data['volatility_regime'].map(lambda x: volatility_weights[x][0])
    data['volatility_weight_5d'] = data['volatility_regime'].map(lambda x: volatility_weights[x][1])
    data['volatility_weight_10d'] = data['volatility_regime'].map(lambda x: volatility_weights[x][2])
    data['volatility_weight_20d'] = data['volatility_regime'].map(lambda x: volatility_weights[x][3])
    
    # Calculate base weights and quality-adjusted weights
    data['base_weight_3d'] = (data['volume_weight_3d'] + data['volatility_weight_3d']) / 2
    data['base_weight_5d'] = (data['volume_weight_5d'] + data['volatility_weight_5d']) / 2
    data['base_weight_10d'] = (data['volume_weight_10d'] + data['volatility_weight_10d']) / 2
    data['base_weight_20d'] = (data['volume_weight_20d'] + data['volatility_weight_20d']) / 2
    
    data['quality_weight_3d'] = data['base_weight_3d'] * data['direction_strength']
    data['quality_weight_5d'] = data['base_weight_5d'] * data['direction_strength']
    data['quality_weight_10d'] = data['base_weight_10d'] * data['direction_strength']
    data['quality_weight_20d'] = data['base_weight_20d'] * data['direction_strength']
    
    # Alpha Signal Generation
    data['raw_score'] = (
        data['quality_weight_3d'] * data['div_3d'] +
        data['quality_weight_5d'] * data['div_5d'] +
        data['quality_weight_10d'] * data['div_10d'] +
        data['quality_weight_20d'] * data['div_20d']
    )
    
    # Signal Enhancement
    data['regime_clarity'] = (abs(data['volume_ratio'] - 1) + abs(data['range_ratio'] - 1)) / 2
    data['enhanced_signal'] = data['raw_score'] * (1 + data['regime_clarity'])
    
    # Return the final alpha factor
    return data['enhanced_signal']
