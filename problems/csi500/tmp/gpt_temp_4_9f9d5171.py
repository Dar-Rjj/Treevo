import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Core Momentum Components
    data['short_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['medium_momentum'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_acceleration'] = data['short_momentum'] - data['medium_momentum']
    
    # Range-Based Momentum Enhancement
    data['daily_range'] = data['high'] - data['low']
    data['range_momentum'] = (data['daily_range'] - data['daily_range'].shift(5)) / data['daily_range'].shift(5)
    data['range_trend'] = data['daily_range'] / data['daily_range'].shift(10)
    
    # Momentum Quality Scoring
    data['price_range_alignment'] = data['momentum_acceleration'] * data['range_trend']
    data['volatility_context'] = data['momentum_acceleration'] / (data['true_range'] / data['true_range'].shift(5))
    data['momentum_quality_score'] = data['price_range_alignment'] * data['volatility_context']
    
    # Volume Dynamics with Pressure Integration
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_ratio'] = data['volume'].rolling(5).mean() / data['volume'].rolling(10).mean()
    data['volume_divergence'] = data['volume_momentum'] - data['volume_ratio']
    
    # Order Flow Pressure Calculation
    data['buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['selling_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    data['net_pressure'] = data['buying_pressure'] - data['selling_pressure']
    
    # Volume Persistence Signals
    data['volume_direction'] = np.sign(data['volume'] - data['volume'].shift(1))
    
    # Calculate consecutive volume days
    data['volume_change_sign'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['consecutive_volume_days'] = 0
    for i in range(1, len(data)):
        if data['volume_change_sign'].iloc[i] == data['volume_change_sign'].iloc[i-1]:
            data['consecutive_volume_days'].iloc[i] = data['consecutive_volume_days'].iloc[i-1] + 1
        else:
            data['consecutive_volume_days'].iloc[i] = 1
    
    data['volume_persistence'] = data['consecutive_volume_days'] * data['volume_direction']
    
    # Efficiency-Weighted Divergence Framework
    data['range_efficiency'] = abs(data['close'] - data['open']) / data['true_range']
    
    # Price-Volume Divergence Analysis
    data['price_volume_divergence'] = data['momentum_acceleration'] - data['volume_momentum']
    data['efficiency_weighted_divergence'] = data['price_volume_divergence'] * data['range_efficiency']
    data['volume_persistence_confirmation'] = data['efficiency_weighted_divergence'] * data['volume_persistence']
    
    # Pressure-Enhanced Divergence
    data['daily_pressure_score'] = data['net_pressure'] * data['volume']
    data['cumulative_pressure'] = data['daily_pressure_score'].rolling(5).sum()
    data['pressure_efficiency'] = data['cumulative_pressure'] / data['true_range']
    data['pressure_enhanced_divergence'] = data['price_volume_divergence'] * data['pressure_efficiency']
    
    # Breakout Conditions with Quality Confirmation
    data['upper_breakout'] = data['close'] > data['high'].rolling(20).max().shift(1)
    data['lower_breakout'] = data['close'] < data['low'].rolling(20).min().shift(1)
    
    # Calculate breakout distance
    data['high_20_max'] = data['high'].rolling(20).max().shift(1)
    data['low_20_min'] = data['low'].rolling(20).min().shift(1)
    data['breakout_distance'] = np.where(
        data['upper_breakout'], 
        abs(data['close'] - data['high_20_max']) / data['true_range'],
        np.where(
            data['lower_breakout'],
            abs(data['close'] - data['low_20_min']) / data['true_range'],
            0
        )
    )
    
    # Quality-Enhanced Breakout Detection
    data['breakout_condition'] = data['upper_breakout'] | data['lower_breakout']
    data['quality_breakout'] = data['breakout_condition'] * data['momentum_quality_score']
    data['efficiency_breakout'] = data['breakout_condition'] * data['range_efficiency']
    data['volume_breakout'] = data['breakout_condition'] * data['volume_persistence']
    
    # Breakout Strength Assessment
    data['combined_quality'] = data['quality_breakout'] + data['efficiency_breakout'] + data['volume_breakout']
    data['pressure_support'] = data['cumulative_pressure'] / data['cumulative_pressure'].rolling(5).mean()
    data['final_breakout_score'] = data['combined_quality'] * data['pressure_support']
    
    # Composite Alpha Signal Generation
    # Core Momentum Quality Factor
    data['quality_weighted_momentum'] = data['momentum_acceleration'] * data['momentum_quality_score']
    data['pressure_enhanced_momentum'] = data['momentum_acceleration'] * data['pressure_efficiency']
    data['core_factor'] = data['quality_weighted_momentum'] + data['pressure_enhanced_momentum']
    
    # Divergence Adjustment Layer
    data['efficiency_weighted_divergence_component'] = data['efficiency_weighted_divergence'] * data['volume_persistence']
    data['divergence_factor'] = data['efficiency_weighted_divergence_component'] + data['pressure_enhanced_divergence']
    
    # Breakout Enhancement
    data['breakout_multiplier'] = 1 + data['final_breakout_score']
    data['directional_breakout'] = data['breakout_multiplier'] * np.sign(data['momentum_acceleration'])
    data['enhanced_momentum'] = data['core_factor'] * data['breakout_multiplier']
    
    # Final Alpha Integration
    data['base_signal'] = data['enhanced_momentum'] + data['divergence_factor']
    data['efficiency_filter'] = data['base_signal'] * data['range_efficiency']
    data['volatility_adjustment'] = data['efficiency_filter'] / (data['true_range'] / data['true_range'].shift(5))
    data['final_alpha'] = data['volatility_adjustment'] * np.sign(data['momentum_acceleration'])
    
    return data['final_alpha']
