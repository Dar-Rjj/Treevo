import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['volatility'] = df['high'] - df['low']
    df['returns'] = df['close'] / df['close'].shift(1) - 1
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['volatility_ratio'] = df['volatility'] / df['volatility'].shift(1)
    
    # Fractal Momentum Components
    df['intraday_fractal'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['overnight_fractal'] = (df['open'] - df['close'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1))
    
    # Fractal Momentum Persistence
    for i in range(len(df)):
        if i >= 2:
            signs = []
            for j in range(i-2, i+1):
                if j >= 1:
                    sign_current = np.sign(df['close'].iloc[j] - df['close'].iloc[j-1])
                    sign_prev = np.sign(df['close'].iloc[j-1] - df['close'].iloc[j-2]) if j >= 2 else 0
                    if sign_prev != 0:
                        signs.append(1 if sign_current == sign_prev else 0)
            df.loc[df.index[i], 'fractal_persistence'] = np.mean(signs) if signs else 0
        else:
            df.loc[df.index[i], 'fractal_persistence'] = 0
    
    # Multi-Scale Fractal Alignment
    df['short_medium_alignment'] = (np.sign(df['returns']) * 
                                   np.sign(df['close'] / df['close'].shift(5) - 1))
    df['fractal_acceleration'] = df['returns'] - df['returns'].shift(1)
    df['alignment_strength'] = (np.abs(df['short_medium_alignment']) * 
                               np.abs(df['fractal_acceleration']))
    
    # Enhanced Fractal Momentum Signals
    df['cross_scale_fractal'] = df['short_medium_alignment'] * df['fractal_acceleration']
    df['persistence_enhanced'] = df['cross_scale_fractal'] * df['fractal_persistence']
    df['dynamic_fractal'] = df['persistence_enhanced'] * df['alignment_strength']
    
    # Fracture-Asymmetry Detection
    df['asymmetric_gap'] = (np.abs(df['returns'] * df['volatility_ratio']) - 
                           np.abs(df['returns'].shift(1) * df['volatility_ratio'].shift(1)))
    df['fracture_intensity'] = df['volatility'] * df['asymmetric_gap']
    
    # Volume-Fractal Momentum Discontinuity
    df['volume_momentum_div'] = df['volume_ratio'] * df['returns']
    df['volume_spike'] = df['volume'] / ((df['volume'].shift(4) + df['volume'].shift(3) + 
                                        df['volume'].shift(2) + df['volume'].shift(1)) / 4)
    
    # Range-Fractal Momentum Dynamics
    df['range_expansion'] = df['volatility_ratio'] * df['returns']
    df['range_efficiency'] = (np.abs(df['close'] - df['open']) / df['volatility']) * df['asymmetric_gap']
    
    # Microstructure-Fractal Integration
    df['volume_efficiency'] = df['intraday_fractal'] * df['volume_ratio']
    df['microstructure_noise'] = df['volatility'] / (df['close'] - df['open'])
    df['efficiency_persistence'] = df['volume_efficiency'] * df['fractal_persistence']
    
    df['fractal_relative_range'] = df['intraday_fractal']
    df['volume_weighted_range'] = df['volatility'] * df['volume_ratio']
    df['microstructure_friction'] = df['microstructure_noise'] * df['volume_weighted_range']
    
    # Fractal Regime Transition
    df['efficiency_regime'] = np.sign(df['volume_efficiency'] - df['volume_efficiency'].shift(1))
    df['friction_regime'] = np.sign(df['microstructure_friction'] - df['microstructure_friction'].shift(1))
    df['regime_momentum'] = df['efficiency_regime'] * df['friction_regime']
    
    # Dynamic Fracture-Asymmetry Integration
    df['volatility_adjusted_momentum'] = df['returns'] / df['volatility']
    df['fracture_volatility_intensity'] = df['asymmetric_gap'] * df['volatility_ratio']
    
    df['range_efficiency_momentum'] = df['range_efficiency'] * df['returns']
    df['volume_efficiency_fracture'] = ((df['close'] - df['close'].shift(1)) / df['volume']) * df['volume_ratio']
    
    # Multi-Scale Fractal Integration
    df['ultra_short_fractal'] = ((df['close'] - df['close'].shift(1)) / 
                                df['volatility'].shift(1))
    df['short_term_fractal'] = df['close'] / df['close'].shift(2) - 1
    df['medium_term_fractal'] = df['close'] / df['close'].shift(5) - 1
    
    df['ultra_short_alignment'] = (np.sign(df['ultra_short_fractal']) * 
                                  np.sign(df['short_term_fractal']))
    df['short_medium_alignment_ms'] = (np.sign(df['short_term_fractal']) * 
                                      np.sign(df['medium_term_fractal']))
    df['multi_scale_consistency'] = df['ultra_short_alignment'] * df['short_medium_alignment_ms']
    
    df['fractal_scale_product'] = (df['ultra_short_fractal'] * df['short_term_fractal'] * 
                                  df['medium_term_fractal'])
    df['consistency_enhanced'] = df['fractal_scale_product'] * df['multi_scale_consistency']
    df['multi_scale_factor'] = df['consistency_enhanced'] * np.abs(df['fractal_scale_product'])
    
    # Fractal Breakout Validation Framework
    for i in range(len(df)):
        if i >= 2:
            # Fractal Asymmetry Persistence
            asym_persistence = sum(1 for j in range(i-2, i+1) if df['volatility_ratio'].iloc[j] > 1) / 3
            
            # Fractal Volume Stability
            vol_stability = sum(1 for j in range(i-2, i+1) if df['volume_ratio'].iloc[j] > 1) / 3
            
            # Fractal Flow Pattern
            flow_pattern = sum(1 for j in range(i-2, i+1) if (df['volume'].iloc[j] > df['volume'].iloc[j-1] and 
                                                             df['volume'].iloc[j-1] > df['volume'].iloc[j-2])) / 3
            
            df.loc[df.index[i], 'asymmetry_persistence'] = asym_persistence
            df.loc[df.index[i], 'volume_stability'] = vol_stability
            df.loc[df.index[i], 'flow_pattern'] = flow_pattern
        else:
            df.loc[df.index[i], 'asymmetry_persistence'] = 0
            df.loc[df.index[i], 'volume_stability'] = 0
            df.loc[df.index[i], 'flow_pattern'] = 0
    
    # Fracture-Asymmetry Signal Validation
    df['price_volume_alignment'] = (np.sign(df['returns'] * df['volatility_ratio']) * 
                                   np.sign(df['volume_ratio']))
    df['range_momentum_coherence'] = (np.sign(df['volatility'] - df['volatility'].shift(1)) * 
                                     np.sign(df['returns'] * df['volatility_ratio']))
    df['volume_fractal_regime'] = np.sign(df['volume_ratio']) * np.sign(df['volatility_ratio'])
    
    # Core Fractal Components
    df['fractal_efficiency_core'] = df['volume_efficiency'] * df['fractal_persistence']
    df['fracture_momentum_core'] = df['dynamic_fractal'] * df['price_volume_alignment']
    df['fractal_momentum_core'] = df['dynamic_fractal'] * df['multi_scale_consistency']
    df['fractal_flow_core'] = df['volume_efficiency'] * df['flow_pattern']
    
    # Fractal Regime Adaptation
    df['high_efficiency_weight'] = np.where(df['volume_efficiency'] > df['volume_efficiency'].shift(1), 1.4, 1.0)
    df['low_friction_weight'] = np.where(df['microstructure_friction'] < df['microstructure_friction'].shift(1), 1.3, 1.0)
    df['fractal_flow_boost'] = np.where((df['fractal_acceleration'] > 0.2) & (df['volume_ratio'] > 1), 1.3, 1.0)
    df['fractal_regime_weight'] = df['high_efficiency_weight'] * df['low_friction_weight'] * df['fractal_flow_boost']
    
    # Validated Fractal Signals
    df['validated_efficiency'] = df['fractal_efficiency_core'] * df['volume_spike']
    df['confirmed_fracture'] = df['fracture_momentum_core'] * df['asymmetry_persistence']
    df['persistent_fractal'] = df['fractal_momentum_core'] * df['fractal_persistence']
    df['confirmed_microstructure'] = df['fractal_flow_core'] * df['asymmetry_persistence']
    
    # Final Alpha Construction
    primary_factor = df['validated_efficiency'] * df['fractal_regime_weight']
    secondary_factor = df['confirmed_fracture'] * df['low_friction_weight']
    tertiary_factor = df['persistent_fractal'] * df['high_efficiency_weight']
    quaternary_factor = df['confirmed_microstructure'] * df['fractal_flow_boost']
    
    # Composite Fracture-Momentum Alpha
    result = (primary_factor * 0.4 + secondary_factor * 0.3 + 
             tertiary_factor * 0.2 + quaternary_factor * 0.1)
    
    # Handle NaN values
    result = result.fillna(0)
    
    return result
