import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Price-Volume Divergence Analysis
    data['price_momentum_divergence'] = ((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) - \
                                       ((data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1))
    
    data['gap_volume_asymmetry'] = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)) * \
                                  np.sign(data['volume'] - data['volume'].shift(1))
    
    data['intraday_volume_pressure'] = (data['high'] - data['low']) * data['volume'] / data['amount']
    
    # Fractal Market Structure
    data['fractal_range_ratio'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['fractal_range_ratio'] = data['fractal_range_ratio'].replace([np.inf, -np.inf], np.nan)
    
    data['price_fractal_dimension'] = np.log(data['high'] - data['low']) / np.log(data['close'] - data['open'])
    data['price_fractal_dimension'] = data['price_fractal_dimension'].replace([np.inf, -np.inf], np.nan)
    
    data['volume_fractal_pattern'] = data['volume'] / np.sqrt(data['volume'].shift(1) * data['volume'].shift(2))
    data['volume_fractal_pattern'] = data['volume_fractal_pattern'].replace([np.inf, -np.inf], np.nan)
    
    # Regime-Dependent Efficiency Metrics
    data['volatility_adjusted_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['volatility_adjusted_momentum'] = data['volatility_adjusted_momentum'].replace([np.inf, -np.inf], np.nan)
    
    data['efficiency_regime_indicator'] = (data['high'] - data['low']) / \
                                         ((data['high'].shift(1) - data['low'].shift(1)) + \
                                          (data['high'].shift(2) - data['low'].shift(2))) * 0.5
    data['efficiency_regime_indicator'] = data['efficiency_regime_indicator'].replace([np.inf, -np.inf], np.nan)
    
    # Regime Adaptation Factors
    data['regime_adaptation_factor'] = 1.0
    data.loc[data['efficiency_regime_indicator'] > 1.3, 'regime_adaptation_factor'] = 0.7
    data.loc[data['efficiency_regime_indicator'] < 0.8, 'regime_adaptation_factor'] = 1.4
    
    # Multi-Scale Momentum Divergence
    data['micro_momentum'] = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    data['macro_momentum'] = (data['close'] - data['close'].shift(8)) / data['close'].shift(8)
    data['momentum_scale_divergence'] = data['micro_momentum'] / data['macro_momentum']
    data['momentum_scale_divergence'] = data['momentum_scale_divergence'].replace([np.inf, -np.inf], np.nan)
    
    # Volume Accumulation Patterns
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) - \
                                 (data['volume'].shift(1) / data['volume'].shift(2))
    
    data['price_volume_coherence'] = np.sign(data['close'] - data['close'].shift(1)) * \
                                    np.sign(data['volume'] - data['volume'].shift(1))
    
    # Accumulation Strength
    data['volume_increase'] = data['volume'] > data['volume'].shift(1)
    data['price_increase'] = data['close'] > data['close'].shift(1)
    data['accumulation_condition'] = data['volume_increase'] & data['price_increase']
    
    # Calculate consecutive accumulation days
    data['accumulation_strength'] = 0
    current_streak = 0
    for i in range(len(data)):
        if data['accumulation_condition'].iloc[i]:
            current_streak += 1
        else:
            current_streak = 0
        data['accumulation_strength'].iloc[i] = current_streak
    
    # Signal Integration Framework
    data['core_divergence_signal'] = data['price_momentum_divergence'] * data['gap_volume_asymmetry']
    data['fractal_enhancement'] = data['fractal_range_ratio'] * data['price_fractal_dimension']
    data['volume_dynamics'] = data['volume_acceleration'] * data['price_volume_coherence']
    data['regime_modulation'] = data['regime_adaptation_factor'] * (1 / data['efficiency_regime_indicator'])
    
    # Final Alpha Construction
    data['primary_factor'] = data['core_divergence_signal'] * data['fractal_enhancement'] * \
                            data['momentum_scale_divergence'] * data['regime_modulation']
    
    data['accumulation_multiplier'] = 1 + (data['accumulation_strength'] / 8)
    
    # Activation Conditions
    activation_condition = (np.abs(data['intraday_volume_pressure']) > 1000) & (data['volume_acceleration'] > 0.1)
    
    # Final alpha factor
    data['alpha_factor'] = data['primary_factor'] * data['accumulation_multiplier']
    data.loc[~activation_condition, 'alpha_factor'] = 0
    
    # Fill NaN values with 0
    data['alpha_factor'] = data['alpha_factor'].fillna(0)
    
    return data['alpha_factor']
