import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Price Structure Analysis
    # Multi-Scale Price Momentum
    data['short_term_momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['long_term_momentum'] = (data['close'] - data['close'].shift(6)) / data['close'].shift(6)
    
    # Fractal Price Patterns
    data['price_acceleration'] = data['short_term_momentum'] - data['medium_term_momentum']
    data['momentum_divergence'] = data['medium_term_momentum'] - data['long_term_momentum']
    data['fractal_consistency'] = np.sign(data['short_term_momentum']) * np.sign(data['medium_term_momentum']) * np.sign(data['long_term_momentum'])
    
    # Price Range Dynamics
    data['intraday_range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['overnight_range_efficiency'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    data['total_range_efficiency'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume Fractal Architecture
    # Volume Momentum Structure
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    data['volume_persistence'] = (data['volume'] / data['volume'].shift(3)) - (data['volume'].shift(3) / data['volume'].shift(6))
    data['volume_momentum_divergence'] = data['volume_acceleration'] - data['volume_persistence']
    
    # Volume Distribution Analysis
    data['volume_concentration'] = data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)).replace(0, np.nan)
    data['volume_dispersion'] = abs(data['volume'] - data['volume'].shift(1)) / (data['volume'] + data['volume'].shift(1)).replace(0, np.nan)
    data['volume_stability'] = 1 - data['volume_dispersion']
    
    # Volume-Price Integration
    data['volume_price_momentum'] = data['short_term_momentum'] * data['volume_acceleration']
    data['volume_price_efficiency'] = data['total_range_efficiency'] * data['volume_concentration']
    data['volume_price_divergence'] = data['momentum_divergence'] * data['volume_momentum_divergence']
    
    # Asymmetric Regime Detection
    # Price Regime Classification
    data['trend_regime'] = (abs(data['short_term_momentum']) > abs(data['medium_term_momentum'])) & (abs(data['medium_term_momentum']) > abs(data['long_term_momentum']))
    data['mean_reversion_regime'] = (abs(data['short_term_momentum']) < abs(data['medium_term_momentum'])) & (abs(data['medium_term_momentum']) < abs(data['long_term_momentum']))
    data['transition_regime'] = ~data['trend_regime'] & ~data['mean_reversion_regime']
    
    # Volume Regime Classification
    data['high_volume_regime'] = (data['volume'] > data['volume'].shift(1)) & (data['volume'] > data['volume'].shift(2)) & (data['volume'] > data['volume'].shift(3))
    data['low_volume_regime'] = (data['volume'] < data['volume'].shift(1)) & (data['volume'] < data['volume'].shift(2)) & (data['volume'] < data['volume'].shift(3))
    data['normal_volume_regime'] = ~data['high_volume_regime'] & ~data['low_volume_regime']
    
    # Efficiency Regime Classification
    data['high_efficiency_regime'] = data['total_range_efficiency'] > 0.7
    data['low_efficiency_regime'] = data['total_range_efficiency'] < 0.3
    data['medium_efficiency_regime'] = ~data['high_efficiency_regime'] & ~data['low_efficiency_regime']
    
    # Fractal Momentum Integration
    # Multi-Scale Momentum Convergence
    data['short_medium_convergence'] = data['short_term_momentum'] * data['medium_term_momentum']
    data['medium_long_convergence'] = data['medium_term_momentum'] * data['long_term_momentum']
    data['full_convergence'] = data['fractal_consistency'] * (data['short_term_momentum'] + data['medium_term_momentum'] + data['long_term_momentum'])
    
    # Volume-Enhanced Momentum
    data['volume_weighted_momentum'] = data['short_term_momentum'] * data['volume_concentration']
    data['acceleration_enhanced_momentum'] = data['price_acceleration'] * data['volume_acceleration']
    data['divergence_enhanced_momentum'] = data['momentum_divergence'] * data['volume_momentum_divergence']
    
    # Efficiency-Adjusted Momentum
    data['range_efficient_momentum'] = data['short_term_momentum'] * data['total_range_efficiency']
    data['intraday_efficient_momentum'] = data['short_term_momentum'] * data['intraday_range_efficiency']
    data['overnight_efficient_momentum'] = data['short_term_momentum'] * data['overnight_range_efficiency']
    
    # Initialize factor column
    data['selected_factor'] = 0.0
    
    # Asymmetric Factor Construction
    # Trend Regime Factors
    trend_high_volume = data['full_convergence'] * data['volume_price_momentum'] * data['total_range_efficiency']
    trend_low_volume = data['acceleration_enhanced_momentum'] * data['volume_stability'] * data['intraday_range_efficiency']
    trend_normal_volume = data['volume_weighted_momentum'] * data['fractal_consistency'] * data['overnight_range_efficiency']
    
    # Mean Reversion Regime Factors
    mr_high_volume = data['divergence_enhanced_momentum'] * data['volume_price_divergence'] * data['volume_dispersion']
    mr_low_volume = data['range_efficient_momentum'] * data['volume_stability'] * data['momentum_divergence']
    mr_normal_volume = data['volume_price_efficiency'] * data['price_acceleration'] * data['volume_persistence']
    
    # Transition Regime Factors
    transition_high_volume = data['short_medium_convergence'] * data['volume_concentration'] * data['intraday_efficient_momentum']
    transition_low_volume = data['medium_long_convergence'] * data['volume_dispersion'] * data['overnight_efficient_momentum']
    transition_normal_volume = data['volume_price_momentum'] * data['volume_price_efficiency'] * data['total_range_efficiency']
    
    # Hierarchical Factor Selection
    # Apply regime-based selection
    for idx in data.index:
        if data.loc[idx, 'trend_regime']:
            if data.loc[idx, 'high_volume_regime']:
                data.loc[idx, 'selected_factor'] = trend_high_volume.loc[idx]
            elif data.loc[idx, 'low_volume_regime']:
                data.loc[idx, 'selected_factor'] = trend_low_volume.loc[idx]
            else:
                data.loc[idx, 'selected_factor'] = trend_normal_volume.loc[idx]
        elif data.loc[idx, 'mean_reversion_regime']:
            if data.loc[idx, 'high_volume_regime']:
                data.loc[idx, 'selected_factor'] = mr_high_volume.loc[idx]
            elif data.loc[idx, 'low_volume_regime']:
                data.loc[idx, 'selected_factor'] = mr_low_volume.loc[idx]
            else:
                data.loc[idx, 'selected_factor'] = mr_normal_volume.loc[idx]
        else:  # Transition regime
            if data.loc[idx, 'high_volume_regime']:
                data.loc[idx, 'selected_factor'] = transition_high_volume.loc[idx]
            elif data.loc[idx, 'low_volume_regime']:
                data.loc[idx, 'selected_factor'] = transition_low_volume.loc[idx]
            else:
                data.loc[idx, 'selected_factor'] = transition_normal_volume.loc[idx]
    
    # Factor Enhancement
    data['volume_momentum_enhancement'] = data['selected_factor'] * (1 + data['volume_acceleration'] * 0.1)
    data['price_momentum_enhancement'] = data['volume_momentum_enhancement'] * (1 + data['price_acceleration'] * 0.1)
    data['efficiency_enhancement'] = data['price_momentum_enhancement'] * (1 + data['total_range_efficiency'] * 0.1)
    
    # Final Factor Output
    data['asymmetric_weighting'] = data['efficiency_enhancement'] * (1 + data['fractal_consistency'] * 0.15)
    data['final_factor'] = data['asymmetric_weighting'] * (1 + np.sign(data['volume_price_momentum']) * 0.1)
    
    # Return the final factor series
    return data['final_factor']
