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
    
    # Nonlinear Momentum Acceleration
    data['quadratic_momentum'] = (data['close'] / data['close'].shift(2))**2 - (data['close'] / data['close'].shift(1))**2
    data['momentum_curvature'] = (data['close'] / data['close'].shift(1)) - (data['close'].shift(1) / data['close'].shift(2))
    data['second_derivative_acceleration'] = data['momentum_curvature'] - data['momentum_curvature'].shift(1)
    
    # Fractal Volume-Pressure Integration
    data['volume_fractal'] = np.log(data['volume']) / np.log(data['volume'].shift(1) + data['volume'].shift(2))
    data['volume_momentum_alignment'] = data['volume_fractal'] * data['momentum_asymmetry']
    
    # Bidirectional Pressure Calculation
    data['buy_side_pressure'] = (data['high'] - data['open']) * data['volume']
    data['sell_side_pressure'] = (data['open'] - data['low']) * data['volume']
    data['net_pressure_ratio'] = (data['buy_side_pressure'] - data['sell_side_pressure']) / (data['buy_side_pressure'] + data['sell_side_pressure'] + 1e-8)
    
    # Pressure Efficiency Metrics
    data['buy_side_efficiency'] = (data['high'] - data['open']) * data['volume'] / (data['high'] - data['low'] + 1e-8)
    data['sell_side_efficiency'] = (data['open'] - data['low']) * data['volume'] / (data['high'] - data['low'] + 1e-8)
    
    # Pressure-Momentum Integration
    data['pressure_momentum_alignment'] = data['net_pressure_ratio'] * data['momentum_curvature']
    data['asymmetric_pressure_acceleration'] = data['net_pressure_ratio'] * data['second_derivative_acceleration']
    
    # Fractal Volatility Regime Classification
    data['price_path_volatility'] = (np.abs(data['high'] - data['close']) + np.abs(data['close'] - data['low'])) / (data['high'] - data['low'] + 1e-8)
    data['volume_volatility'] = data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3) + 1e-8)
    data['fractal_volatility_regime'] = data['price_path_volatility'] * data['volume_volatility']
    
    # Asymmetric Breakout Detection
    data['price_breakout'] = data['close'] > pd.concat([data['high'].shift(1), data['high'].shift(2), data['high'].shift(3)], axis=1).max(axis=1)
    data['volume_surge'] = data['volume'] > 1.5 * (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3
    data['breakout_asymmetry'] = (data['buy_side_pressure'] - data['sell_side_pressure']) / (data['high'] - data['low'] + 1e-8)
    
    # Regime-Specific Momentum Adjustment
    data['high_vol_momentum'] = data['quadratic_momentum'] / (data['fractal_volatility_regime'] + 1e-8)
    data['low_vol_acceleration'] = data['second_derivative_acceleration'] * data['fractal_volatility_regime']
    
    # Multi-Scale Asymmetry Alignment
    data['micro_macro_coherence'] = data['short_term_momentum'] / (data['medium_term_momentum'] + 1e-8)
    data['acceleration_consistency'] = data['quadratic_momentum'] / (data['momentum_curvature'] + 1e-8)
    
    # Volume-Pressure Asymmetry Patterns
    data['volume_weighted_acceleration'] = data['volume_fractal'] * data['quadratic_momentum']
    data['pressure_momentum_correlation'] = data['net_pressure_ratio'] * data['momentum_asymmetry']
    
    # Breakout-Asymmetry Synchronization
    data['breakout_enhanced_asymmetry'] = data['breakout_asymmetry'] * data['momentum_asymmetry']
    data['pressure_confirmed_acceleration'] = data['net_pressure_ratio'] * data['second_derivative_acceleration']
    
    # Regime Detection
    data['is_breakout_regime'] = data['price_breakout'] & data['volume_surge']
    data['is_high_vol_regime'] = data['fractal_volatility_regime'] > data['fractal_volatility_regime'].rolling(20).mean()
    data['is_low_vol_regime'] = data['fractal_volatility_regime'] < data['fractal_volatility_regime'].rolling(20).quantile(0.3)
    
    # Regime-based weighting and combination
    alpha_factor = pd.Series(index=data.index, dtype=float)
    
    for idx in data.index:
        if data.loc[idx, 'is_breakout_regime']:
            # Breakout regime weights
            alpha_factor.loc[idx] = (
                0.4 * data.loc[idx, 'momentum_asymmetry'] +
                0.3 * data.loc[idx, 'breakout_asymmetry'] +
                0.2 * data.loc[idx, 'quadratic_momentum'] +
                0.1 * data.loc[idx, 'net_pressure_ratio']
            )
        elif data.loc[idx, 'is_high_vol_regime']:
            # High volatility regime weights
            alpha_factor.loc[idx] = (
                0.4 * data.loc[idx, 'momentum_asymmetry'] +
                0.3 * data.loc[idx, 'breakout_asymmetry'] +
                0.2 * data.loc[idx, 'quadratic_momentum'] +
                0.1 * data.loc[idx, 'net_pressure_ratio']
            )
        elif data.loc[idx, 'is_low_vol_regime']:
            # Low volatility regime weights
            alpha_factor.loc[idx] = (
                0.3 * data.loc[idx, 'momentum_asymmetry'] +
                0.2 * data.loc[idx, 'breakout_asymmetry'] +
                0.4 * data.loc[idx, 'second_derivative_acceleration'] +
                0.1 * data.loc[idx, 'volume_fractal']
            )
        else:
            # Normal volatility regime weights
            alpha_factor.loc[idx] = (
                0.35 * data.loc[idx, 'momentum_asymmetry'] +
                0.35 * data.loc[idx, 'breakout_asymmetry'] +
                0.2 * data.loc[idx, 'net_pressure_ratio'] +
                0.1 * data.loc[idx, 'momentum_curvature']
            )
    
    # Apply volume fractal confirmation multiplier
    volume_confirmation = 1 + 0.5 * np.tanh(data['volume_fractal'])
    # Apply pressure asymmetry adjustment factor
    pressure_adjustment = 1 + 0.3 * np.tanh(data['net_pressure_ratio'])
    
    # Final composite factor with dynamic refinement
    alpha_factor = alpha_factor * volume_confirmation * pressure_adjustment
    
    # Smooth transitions between regimes using rolling mean
    alpha_factor = alpha_factor.rolling(window=3, min_periods=1).mean()
    
    return alpha_factor
