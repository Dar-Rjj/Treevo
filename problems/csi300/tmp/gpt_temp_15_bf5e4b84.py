import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Momentum Divergence
    data['short_term_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    data['momentum_asymmetry'] = np.sign(data['short_term_momentum']) * (data['short_term_momentum'] - data['medium_term_momentum'])
    data['asymmetric_momentum_response'] = (data['high'] - data['close'].shift(1)) / (data['close'].shift(1) - data['low'])
    
    # Nonlinear Momentum Acceleration
    data['quadratic_momentum'] = (data['close'] / data['close'].shift(2))**2 - (data['close'] / data['close'].shift(1))**2
    data['momentum_curvature'] = (data['close'] - 2 * data['close'].shift(1) + data['close'].shift(2)) / data['close'].shift(2)
    data['second_derivative_acceleration'] = (data['close'] / data['close'].shift(1)) - (data['close'].shift(1) / data['close'].shift(2))
    data['second_derivative_acceleration'] = data['second_derivative_acceleration'] - data['second_derivative_acceleration'].shift(1)
    data['decay_adjusted_momentum'] = data['quadratic_momentum'] * data['second_derivative_acceleration']
    
    # Fractal Volume-Pressure Integration
    data['volume_fractal'] = np.log(data['volume']) / np.log(data['volume'].shift(1) + data['volume'].shift(2))
    data['fracture_volume_ratio'] = data['volume'] / (data['high'] - data['low'])
    data['volume_momentum_alignment'] = data['volume_fractal'] * data['momentum_asymmetry']
    data['volume_fractal_acceleration'] = data['momentum_curvature'] * (data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)))
    
    # Bidirectional Pressure Calculation
    data['buy_side_pressure'] = (data['high'] - data['open']) * data['volume']
    data['sell_side_pressure'] = (data['open'] - data['low']) * data['volume']
    data['net_pressure_ratio'] = (data['buy_side_pressure'] - data['sell_side_pressure']) / (data['buy_side_pressure'] + data['sell_side_pressure'])
    data['pressure_asymmetry_ratio'] = data['buy_side_pressure'] / data['sell_side_pressure']
    
    # Pressure Efficiency Metrics
    data['buy_side_pressure_efficiency'] = (data['high'] - data['open']) * data['volume'] / (data['high'] - data['low'])
    data['sell_side_pressure_efficiency'] = (data['open'] - data['low']) * data['volume'] / (data['high'] - data['low'])
    data['efficiency_asymmetry'] = data['buy_side_pressure_efficiency'] / data['sell_side_pressure_efficiency']
    
    # Pressure-Momentum Integration
    data['pressure_momentum_alignment'] = data['net_pressure_ratio'] * data['momentum_curvature']
    data['asymmetric_pressure_acceleration'] = data['pressure_asymmetry_ratio'] * data['second_derivative_acceleration']
    data['volume_fractal_pressure'] = data['volume_fractal'] * data['pressure_asymmetry_ratio']
    
    # Fractal Volatility Regime Classification
    data['price_path_volatility'] = (abs(data['high'] - data['close']) + abs(data['close'] - data['low'])) / (data['high'] - data['low'])
    data['volume_volatility'] = data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3))
    data['fractal_volatility_regime'] = data['price_path_volatility'] * data['volume_volatility']
    
    # Volatility regime classification
    data['avg_range_20'] = (data['high'] - data['low']).rolling(window=20).mean()
    data['high_volatility'] = (data['high'] - data['low']) > 1.5 * data['avg_range_20']
    data['low_volatility'] = (data['high'] - data['low']) < 0.67 * data['avg_range_20']
    data['normal_volatility'] = ~(data['high_volatility'] | data['low_volatility'])
    
    # Asymmetric Breakout Detection
    data['price_breakout'] = data['close'] > pd.concat([data['high'].shift(1), data['high'].shift(2), data['high'].shift(3)], axis=1).max(axis=1)
    data['volume_surge'] = data['volume'] > 1.5 * (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3
    data['breakout_asymmetry'] = (data['buy_side_pressure'] - data['sell_side_pressure']) / (data['high'] - data['low'])
    data['fractal_compression'] = (data['high'] - data['low']) / ((data['high'].shift(1) - data['low'].shift(1)) + (data['high'].shift(2) - data['low'].shift(2)) + (data['high'].shift(3) - data['low'].shift(3))) / 3
    
    # Regime-Specific Momentum Adjustment
    data['high_volatility_momentum'] = data['quadratic_momentum'] / data['fractal_volatility_regime']
    data['low_volatility_acceleration'] = data['second_derivative_acceleration'] * data['fractal_volatility_regime']
    data['transition_momentum'] = data['momentum_curvature'] * abs(data['fractal_volatility_regime'] - data['fractal_volatility_regime'].shift(1))
    data['volatility_pressure_efficiency'] = data['net_pressure_ratio'] / data['fractal_volatility_regime']
    
    # Multi-Scale Asymmetry Alignment
    data['micro_macro_momentum_coherence'] = data['short_term_momentum'] / data['medium_term_momentum']
    data['momentum_fractal_divergence'] = (data['short_term_momentum'] - data['asymmetric_momentum_response']) - (data['asymmetric_momentum_response'] - data['quadratic_momentum'])
    data['acceleration_consistency'] = data['quadratic_momentum'] / data['momentum_curvature']
    
    # Volume-Pressure Asymmetry Patterns
    data['volume_weighted_acceleration'] = data['volume_fractal'] * data['quadratic_momentum']
    data['pressure_momentum_correlation'] = data['pressure_asymmetry_ratio'] * data['momentum_asymmetry']
    data['fracture_confirmed_momentum'] = data['fracture_volume_ratio'] * data['momentum_asymmetry'] / (data['high'] - data['low'])
    data['pressure_enhanced_momentum'] = data['efficiency_asymmetry'] * data['decay_adjusted_momentum']
    
    # Breakout-Asymmetry Synchronization
    data['breakout_enhanced_asymmetry'] = data['breakout_asymmetry'] * data['momentum_asymmetry']
    data['pressure_confirmed_acceleration'] = data['pressure_asymmetry_ratio'] * data['second_derivative_acceleration']
    data['fractal_breakout_momentum'] = data['fractal_compression'] * data['quadratic_momentum']
    data['volume_pressure_breakout'] = data['volume_surge'] * data['net_pressure_ratio']
    
    # Breakout Condition Signals
    data['primary_breakout'] = data['price_breakout'] * data['volume_surge']
    data['pressure_breakout'] = data['breakout_asymmetry'] * data['net_pressure_ratio']
    data['fractal_breakout'] = data['fractal_breakout_momentum'] * data['volume_fractal_pressure']
    data['asymmetric_breakout'] = data['breakout_enhanced_asymmetry'] * data['pressure_confirmed_acceleration']
    
    # Momentum Regime Signals
    data['acceleration_momentum'] = data['decay_adjusted_momentum'] * data['second_derivative_acceleration']
    data['pressure_momentum'] = data['pressure_momentum_alignment'] * data['asymmetric_pressure_acceleration']
    data['fractal_momentum'] = data['volume_weighted_acceleration'] * data['volume_fractal_acceleration']
    data['asymmetric_momentum'] = data['momentum_asymmetry'] * data['asymmetric_momentum_response']
    
    # Volatility Adjustment
    data['high_volatility_alpha'] = data['high_volatility_momentum'] * data['volatility_pressure_efficiency']
    data['low_volatility_alpha'] = data['low_volatility_acceleration'] * data['pressure_enhanced_momentum']
    data['transition_alpha'] = data['transition_momentum'] * data['fracture_confirmed_momentum']
    
    # Final Composite Alpha Factor
    alpha_factor = pd.Series(index=data.index, dtype=float)
    
    for idx in data.index:
        if data.loc[idx, 'price_breakout'] and data.loc[idx, 'volume_surge']:
            # Breakout regime
            breakout_signal = (data.loc[idx, 'primary_breakout'] + data.loc[idx, 'pressure_breakout'] + 
                             data.loc[idx, 'fractal_breakout'] + data.loc[idx, 'asymmetric_breakout']) / 4
            alpha_factor.loc[idx] = breakout_signal
        elif data.loc[idx, 'high_volatility']:
            # High volatility regime
            alpha_factor.loc[idx] = data.loc[idx, 'high_volatility_alpha']
        elif data.loc[idx, 'low_volatility']:
            # Low volatility regime
            alpha_factor.loc[idx] = data.loc[idx, 'low_volatility_alpha']
        else:
            # Default regime - Momentum Regime Signals
            momentum_signal = (data.loc[idx, 'acceleration_momentum'] + data.loc[idx, 'pressure_momentum'] + 
                             data.loc[idx, 'fractal_momentum'] + data.loc[idx, 'asymmetric_momentum']) / 4
            alpha_factor.loc[idx] = momentum_signal
    
    # Apply fractal structure enhancements
    alpha_factor = alpha_factor * data['momentum_fractal_divergence']
    alpha_factor = alpha_factor * data['acceleration_consistency']
    alpha_factor = alpha_factor / data['fractal_volatility_regime']
    
    return alpha_factor
