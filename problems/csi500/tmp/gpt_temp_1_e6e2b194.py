import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Momentum-Volume Regime Factor
    Combines multi-timeframe momentum with volume confirmation and volatility regime adaptation
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate multi-timeframe momentum
    data['momentum_1d'] = data['close'] - data['close'].shift(1)
    data['momentum_3d'] = data['close'] - data['close'].shift(2)
    data['momentum_5d'] = data['close'] - data['close'].shift(4)
    data['momentum_10d'] = data['close'] - data['close'].shift(9)
    
    # Momentum quality assessment
    momentum_cols = ['momentum_1d', 'momentum_3d', 'momentum_5d', 'momentum_10d']
    data['direction_consistency'] = data[momentum_cols].apply(
        lambda x: sum(np.sign(x) == np.sign(x['momentum_1d'])) if not pd.isna(x['momentum_1d']) else 0, axis=1
    )
    
    # Momentum persistence (consecutive days with same direction)
    data['momentum_direction'] = np.sign(data['momentum_1d'])
    data['momentum_persistence'] = (data['momentum_direction'] == data['momentum_direction'].shift(1)).astype(int)
    data['momentum_persistence'] = data['momentum_persistence'].groupby((data['momentum_persistence'] != data['momentum_persistence'].shift()).cumsum()).cumsum()
    
    # Acceleration
    data['acceleration'] = data['momentum_1d'] - data['momentum_1d'].shift(1)
    
    # Momentum strength measures
    data['abs_momentum'] = data[momentum_cols].abs().mean(axis=1)
    data['momentum_ratio'] = data['momentum_3d'] / data['momentum_5d'].replace(0, np.nan)
    data['momentum_stability'] = data[momentum_cols].std(axis=1)
    
    # Volume dynamics
    data['volume_change'] = data['volume'] / data['volume'].shift(1)
    data['volume_trend'] = data['volume'] / data['volume'].shift(4)
    data['volume_intensity'] = data['amount'] / data['volume'].replace(0, np.nan)
    
    # Volume-momentum alignment
    data['direction_match'] = np.sign(data['momentum_3d']) * np.sign(data['volume_change'] - 1)
    data['alignment_persistence'] = (data['direction_match'] > 0).astype(int)
    data['alignment_persistence'] = data['alignment_persistence'].groupby((data['alignment_persistence'] != data['alignment_persistence'].shift()).cumsum()).cumsum()
    data['alignment_strength'] = data['alignment_persistence'] * abs(data['momentum_3d'])
    
    # Volume regime classification
    data['volume_regime'] = 'normal'
    high_vol_threshold = data['volume'].shift(4) * 1.3
    low_vol_threshold = data['volume'].shift(4) * 0.7
    data.loc[data['volume'] > high_vol_threshold, 'volume_regime'] = 'high'
    data.loc[data['volume'] < low_vol_threshold, 'volume_regime'] = 'low'
    
    # Volatility context
    data['current_range'] = data['high'] - data['low']
    data['short_volatility'] = data['current_range'].rolling(window=3, min_periods=1).mean()
    data['medium_volatility'] = data['current_range'].rolling(window=10, min_periods=1).mean()
    
    # Volatility regime
    data['volatility_regime'] = 'normal'
    high_vol_threshold = data['medium_volatility'] * 1.25
    low_vol_threshold = data['medium_volatility'] * 0.75
    data.loc[data['short_volatility'] > high_vol_threshold, 'volatility_regime'] = 'high'
    data.loc[data['short_volatility'] < low_vol_threshold, 'volatility_regime'] = 'low'
    
    # Volatility trend and persistence
    data['volatility_direction'] = data['short_volatility'] / data['medium_volatility'].replace(0, np.nan)
    data['volatility_persistence'] = (data['volatility_regime'] == data['volatility_regime'].shift(1)).astype(int)
    data['volatility_persistence'] = data['volatility_persistence'].groupby((data['volatility_persistence'] != data['volatility_persistence'].shift()).cumsum()).cumsum()
    data['volatility_stability'] = abs(data['volatility_direction'] - 1)
    
    # Regime-adaptive weighting - Volatility-based timeframe selection
    def get_momentum_weights(regime):
        if regime == 'high':
            return {'primary': 'momentum_1d', 'secondary': 'momentum_3d', 'primary_w': 0.7, 'secondary_w': 0.3}
        elif regime == 'low':
            return {'primary': 'momentum_5d', 'secondary': 'momentum_10d', 'primary_w': 0.55, 'secondary_w': 0.45}
        else:  # normal
            return {'primary': 'momentum_3d', 'secondary': 'momentum_5d', 'primary_w': 0.6, 'secondary_w': 0.4}
    
    # Volume-based enhancement weights
    def get_volume_weights(regime):
        if regime == 'high':
            return {'momentum_w': 0.8, 'volume_w': 0.2, 'alignment_mult': 1.2}
        elif regime == 'low':
            return {'momentum_w': 0.6, 'volume_w': 0.4, 'alignment_mult': 0.8}
        else:  # normal
            return {'momentum_w': 0.7, 'volume_w': 0.3, 'alignment_mult': 1.0}
    
    # Apply regime-based weighting
    data['regime_momentum'] = 0.0
    data['volume_weights'] = 0.0
    data['alignment_multiplier'] = 1.0
    
    for idx, row in data.iterrows():
        if pd.notna(row['volatility_regime']):
            mom_weights = get_momentum_weights(row['volatility_regime'])
            vol_weights = get_volume_weights(row['volume_regime'])
            
            primary_mom = row[mom_weights['primary']] if pd.notna(row[mom_weights['primary']]) else 0
            secondary_mom = row[mom_weights['secondary']] if pd.notna(row[mom_weights['secondary']]) else 0
            
            data.loc[idx, 'regime_momentum'] = (primary_mom * mom_weights['primary_w'] + 
                                              secondary_mom * mom_weights['secondary_w'])
            data.loc[idx, 'momentum_weight'] = vol_weights['momentum_w']
            data.loc[idx, 'volume_weight'] = vol_weights['volume_w']
            data.loc[idx, 'alignment_multiplier'] = vol_weights['alignment_mult']
    
    # Persistence-based scaling
    data['momentum_persistence_factor'] = np.clip(1.0 + data['momentum_persistence'] * 0.05, 1.0, 2.0)
    data['volume_alignment_factor'] = np.clip(1.0 + data['alignment_persistence'] * 0.08, 1.0, 1.8)
    
    # Signal construction
    # Core momentum component
    data['core_momentum'] = (data['regime_momentum'] * data['momentum_persistence_factor'] * 
                           (data['direction_consistency'] / 4))
    
    # Volume confirmation component
    data['volume_score'] = data['volume_change'] * data['volume_weight']
    data['alignment_boost'] = data['volume_score'] * data['alignment_multiplier']
    data['volume_component'] = data['alignment_boost'] * data['volume_alignment_factor']
    
    # Factor integration
    data['base_factor'] = (data['core_momentum'] * data['momentum_weight'] + 
                          data['volume_component'])
    
    # Volatility adjustment
    data['volatility_adjustment'] = 1.0
    data.loc[data['volatility_regime'] == 'high', 'volatility_adjustment'] = 0.9
    data.loc[data['volatility_regime'] == 'low', 'volatility_adjustment'] = 1.1
    
    # Final alpha factor
    data['final_alpha'] = data['base_factor'] * data['volatility_adjustment']
    
    return data['final_alpha']
