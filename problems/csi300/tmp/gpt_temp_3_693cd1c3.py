import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    data['prev_2_volume'] = data['volume'].shift(2)
    data['prev_3_volume'] = data['volume'].shift(3)
    data['prev_4_volume'] = data['volume'].shift(4)
    data['prev_5_volume'] = data['volume'].shift(5)
    data['prev_10_volume'] = data['volume'].shift(10)
    
    # Calculate VWAP-like efficiency measures
    data['efficiency'] = data['amount'] / data['volume']
    data['prev_efficiency'] = data['efficiency'].shift(1)
    data['prev_3_efficiency'] = data['efficiency'].shift(3)
    data['prev_4_efficiency'] = data['efficiency'].shift(4)
    data['prev_5_efficiency'] = data['efficiency'].shift(5)
    data['prev_10_efficiency'] = data['efficiency'].shift(10)
    
    # Microstructure Momentum Components
    # Price-Microstructure Momentum
    data['opening_pressure_momentum'] = ((data['open'] - data['prev_close']) / data['prev_close']) * (data['volume'] / data['prev_volume'])
    data['intraday_efficiency_momentum'] = ((data['close'] - data['open']) / data['open']) * (data['efficiency'] / data['prev_efficiency'])
    data['range_pressure_convergence'] = ((data['high'] - data['low']) / data['prev_close']) * np.sign(data['close'] - data['prev_close'])
    
    # Volume-Microstructure Momentum
    data['volume_efficiency_gradient'] = (data['efficiency'] - data['prev_efficiency']) * (data['volume'] / data['prev_volume'])
    data['volume_concentration_momentum'] = (data['volume'] / (data['prev_volume'] + data['prev_2_volume'])) * ((data['volume'] - data['prev_volume']) / data['prev_volume'])
    data['micro_volume_structure'] = data['volume_efficiency_gradient'] * data['volume_concentration_momentum']
    
    # Microstructure Integration
    data['price_volume_momentum_alignment'] = data['opening_pressure_momentum'] * data['volume_efficiency_gradient']
    data['efficiency_pressure_convergence'] = data['intraday_efficiency_momentum'] * data['range_pressure_convergence']
    data['microstructure_momentum_signal'] = data['price_volume_momentum_alignment'] * data['efficiency_pressure_convergence']
    
    # Multi-Timeframe Momentum Integration
    # Short-term Momentum
    data['price_momentum_shift'] = ((data['close'] - data['prev_close']) / data['prev_close']) - ((data['prev_close'] - data['close'].shift(2)) / data['close'].shift(2))
    data['volume_momentum_acceleration'] = (data['volume'] / data['prev_volume']) * ((data['volume'] - data['prev_volume']) / data['prev_volume'])
    data['efficiency_momentum_change'] = (data['efficiency'] / data['prev_efficiency']) * ((data['efficiency'] - data['prev_efficiency']) / data['prev_efficiency'])
    
    # Medium-term Momentum
    # Price momentum persistence (consecutive days count)
    data['price_up'] = (data['close'] > data['prev_close']).astype(int)
    data['price_down'] = (data['close'] < data['prev_close']).astype(int)
    
    up_count = data['price_up'].rolling(window=5, min_periods=1).apply(lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == 1 and x.iloc[i-1] == 1]), raw=False)
    down_count = data['price_down'].rolling(window=5, min_periods=1).apply(lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == 1 and x.iloc[i-1] == 1]), raw=False)
    data['price_momentum_persistence'] = up_count - down_count
    
    data['volume_momentum_regime'] = (data['volume'] / data['prev_3_volume']) * (data['prev_volume'] / data['prev_4_volume'])
    data['efficiency_momentum_regime'] = (data['efficiency'] / data['prev_3_efficiency']) * (data['prev_efficiency'] / data['prev_4_efficiency'])
    
    # Long-term Momentum
    data['high_low_range'] = data['high'] - data['low']
    data['prev_5_high_low'] = data['high_low_range'].shift(5)
    data['prev_10_high_low'] = data['high_low_range'].shift(10)
    
    data['range_volatility_momentum'] = (data['high_low_range'] / data['prev_5_high_low']) * (data['prev_5_high_low'] / data['prev_10_high_low'])
    data['volume_structure_momentum'] = (data['volume'] / data['prev_5_volume']) * (data['prev_5_volume'] / data['prev_10_volume'])
    data['efficiency_structure_momentum'] = (data['efficiency'] / data['prev_5_efficiency']) * (data['prev_5_efficiency'] / data['prev_10_efficiency'])
    
    # Multi-Scale Integration
    data['short_medium_alignment'] = data['price_momentum_shift'] * data['price_momentum_persistence']
    data['medium_long_convergence'] = data['volume_momentum_regime'] * data['volume_structure_momentum']
    data['multi_scale_momentum_signal'] = data['short_medium_alignment'] * data['medium_long_convergence']
    
    # Regime Detection and Switching
    # Microstructure Regime Patterns
    data['high_pressure_regime'] = (np.abs(data['opening_pressure_momentum']) > np.abs(data['intraday_efficiency_momentum'])).astype(int)
    data['volume_efficiency_regime'] = (data['volume_efficiency_gradient'] > data['volume_concentration_momentum']).astype(int)
    data['microstructure_regime_strength'] = data['microstructure_momentum_signal'] * (1 + np.abs(data['price_volume_momentum_alignment']))
    
    # Momentum Volatility Patterns
    data['momentum_acceleration'] = data['price_momentum_shift'] * data['volume_momentum_acceleration']
    data['volatility_momentum'] = data['range_volatility_momentum'] * data['efficiency_momentum_change']
    data['momentum_volatility_signal'] = data['momentum_acceleration'] * data['volatility_momentum']
    
    # Structural Break Detection
    data['price_structure_shift'] = ((data['close'] - data['prev_close']) / data['prev_close']) - ((data['prev_close'] - data['close'].shift(2)) / data['close'].shift(2))
    data['volume_structure_change'] = (data['volume'] / data['prev_volume']) - (data['prev_volume'] / data['prev_2_volume'])
    data['efficiency_structure_break'] = (data['efficiency'] / data['prev_efficiency']) - (data['prev_efficiency'] / data['efficiency'].shift(2))
    data['structural_break_detection'] = np.abs(data['price_structure_shift']) + np.abs(data['volume_structure_change']) + np.abs(data['efficiency_structure_break'])
    
    # Regime Integration
    data['microstructure_momentum_alignment'] = data['microstructure_regime_strength'] * data['momentum_volatility_signal']
    
    # Structure-Guided Switching
    data['structure_guided_switching'] = np.where(
        np.abs(data['structural_break_detection']) > 0,
        data['microstructure_momentum_alignment'],
        data['momentum_volatility_signal']
    )
    
    data['regime_adaptive_signal'] = data['structure_guided_switching'] * (1 + np.abs(data['microstructure_regime_strength']))
    
    # Dynamic Factor Construction
    # Core Momentum Components
    data['microstructure_driven_momentum'] = data['microstructure_momentum_signal'] * data['multi_scale_momentum_signal']
    data['volatility_enhanced_momentum'] = data['momentum_volatility_signal'] * data['range_volatility_momentum']
    data['structure_confirmed_momentum'] = data['structural_break_detection'] * data['price_momentum_persistence']
    
    # Adaptive Enhancement
    data['microstructure_amplification'] = (
        data['microstructure_driven_momentum'] + 
        data['volatility_enhanced_momentum'] + 
        data['structure_confirmed_momentum']
    ) * (1 + np.abs(data['microstructure_regime_strength']))
    
    data['volatility_modulation'] = data['microstructure_amplification'] * data['momentum_volatility_signal']
    data['structure_validation'] = data['volatility_modulation'] * (1 + np.abs(data['structural_break_detection']))
    
    # Multi-Regime Integration
    data['high_pressure_mode'] = np.where(
        data['high_pressure_regime'] == 1,
        data['microstructure_driven_momentum'] * 2,
        data['structure_validation']
    )
    
    data['efficiency_mode'] = np.where(
        data['volume_efficiency_regime'] == 1,
        data['volatility_enhanced_momentum'] * 1.5,
        data['structure_confirmed_momentum']
    )
    
    # Adaptive Factor Selection
    data['adaptive_factor_selection'] = (
        data['high_pressure_mode'] * data['regime_adaptive_signal'] + 
        data['efficiency_mode'] * (1 - data['regime_adaptive_signal'])
    )
    
    # Final Alpha Factor
    data['primary_signal'] = data['adaptive_factor_selection'] * data['regime_adaptive_signal']
    data['momentum_confirmation'] = data['primary_signal'] * data['multi_scale_momentum_signal']
    data['microstructure_validation'] = data['momentum_confirmation'] * data['microstructure_momentum_signal']
    
    # Multi-Scale Momentum-Microstructure Factor
    alpha_factor = data['microstructure_validation']
    
    return alpha_factor
