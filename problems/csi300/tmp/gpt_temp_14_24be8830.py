import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Basic calculations
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = (data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8)
    data['intraday_range'] = data['high'] - data['low']
    data['close_to_mid'] = data['close'] - (data['high'] + data['low']) / 2
    data['close_to_open'] = data['close'] - data['open']
    data['vwap'] = data['amount'] / (data['volume'] + 1e-8)
    
    # Fractal components
    data['fractal_overnight_gap'] = np.tanh(data['overnight_gap'] * 10)
    data['fractal_gap_persistence'] = np.arctan(data['overnight_gap'].rolling(3).mean() * 5)
    data['fractal_gap_momentum'] = np.tanh(data['overnight_gap'].diff(2) * 20)
    data['fractal_gap_fill_ratio'] = np.arctan(data['close_to_open'] / (abs(data['overnight_gap']) + 1e-8) * 2)
    
    # Quantum Gap Asymmetry Dynamics
    data['gap_opening_quantum_asymmetry'] = (
        (data['open'] - data['prev_close']) * data['volume'] / 
        (data['intraday_range'] + 1e-8) * np.sign(data['close'] - data['prev_close']) * 
        data['fractal_overnight_gap']
    )
    
    data['gap_closing_quantum_asymmetry'] = (
        data['close_to_mid'] * data['volume'] / (data['intraday_range'] + 1e-8) * 
        np.sign(data['close_to_open']) * data['fractal_gap_persistence']
    )
    
    data['gap_volatility_quantum_asymmetry'] = (
        (data['high'] - data['close']) / (data['close'] - data['low'] + 1e-8) * 
        data['volume'] * data['fractal_gap_momentum']
    )
    
    # Gap-Fractal Efficiency & Entropy Integration
    data['gap_quantum_efficiency_ratio'] = (
        (data['close'] - data['close'].shift(5)) / 
        (abs(data['close'].diff()).rolling(5).sum() + 1e-8) * 
        data['volume'] * data['fractal_gap_fill_ratio']
    )
    
    data['gap_quantum_noise_ratio'] = (
        data['intraday_range'] / (abs(data['close_to_open']) + 1e-8) * 
        data['volume'] * data['fractal_overnight_gap']
    )
    
    upward_pressure = data['close'].rolling(5).apply(lambda x: x[x > x.shift(1)].count() if len(x) == 5 else np.nan)
    downward_pressure = data['close'].rolling(5).apply(lambda x: x[x < x.shift(1)].count() if len(x) == 5 else np.nan)
    data['gap_quantum_entropy_pressure'] = (
        (upward_pressure - downward_pressure) / 
        (abs(upward_pressure) + abs(downward_pressure) + 1e-8) * 
        data['fractal_gap_persistence']
    )
    
    # Gap-Fracture-Liquidity Dynamics
    data['gap_fracture_liquidity_momentum'] = (
        data['close_to_open'] / (data['vwap'] + 1e-8) * 
        data['close_to_open'] / (data['intraday_range'] + 1e-8) * 
        np.sign(data['close_to_open']) * data['fractal_overnight_gap']
    )
    
    data['gap_quantum_volume_liquidity'] = (
        (data['volume'] / (data['volume'].shift(1) + 1e-8)) * 
        (data['vwap'] / (data['vwap'].shift(1) + 1e-8)) * 
        (data['close_to_open'] / (data['intraday_range'] + 1e-8)) * 
        data['fractal_gap_momentum']
    )
    
    # Gap-Fractal Regime Detection
    rolling_range_median = data['intraday_range'].rolling(20).median()
    data['gap_high_volatility_fractal'] = (
        (data['intraday_range'] > rolling_range_median * 1.5) * 
        data['fractal_gap_momentum']
    )
    
    data['gap_low_volatility_fractal'] = (
        (data['intraday_range'] < rolling_range_median * 0.7) * 
        data['fractal_gap_persistence']
    )
    
    prev_range_ratio = data['intraday_range'] / (data['intraday_range'].shift(1) + 1e-8)
    data['gap_fractal_regime_shift'] = (
        ((prev_range_ratio > 1.3) | (prev_range_ratio < 0.8)) * 
        data['fractal_overnight_gap']
    )
    
    # Gap-Regime-Adaptive Factor Construction
    data['gap_high_volatility_factor'] = (
        data['gap_volatility_quantum_asymmetry'] * 
        data['gap_quantum_efficiency_ratio'] * 
        data['gap_quantum_volume_liquidity']
    )
    
    data['gap_low_volatility_factor'] = (
        data['gap_closing_quantum_asymmetry'] * 
        data['gap_quantum_entropy_pressure'] * 
        data['gap_fracture_liquidity_momentum']
    )
    
    data['gap_transition_regime_factor'] = (
        data['gap_opening_quantum_asymmetry'] * 
        data['gap_quantum_noise_ratio'] * 
        data['gap_quantum_volume_liquidity']
    )
    
    # Gap-Dynamic Signal Synthesis
    data['gap_regime_adaptive_core'] = (
        data['gap_high_volatility_factor'].fillna(0) * data['gap_fractal_regime_shift'] * 
        (data['close_to_open'] / (data['close'] + 1e-8))
    )
    
    data['gap_quantum_efficiency_core'] = (
        data['gap_transition_regime_factor'] * data['gap_quantum_entropy_pressure'] * 
        (data['volume'] / data['volume'].rolling(10).mean())
    )
    
    # Gap-Fractal Divergence Acceleration
    data['gap_momentum_dispersion_divergence'] = (
        abs(data['close'].diff() / (data['intraday_range'] + 1e-8) - 
           (data['volume'] / data['volume'].shift(1) - 1)) * 
        data['volume'] * data['fractal_gap_persistence']
    )
    
    # Final Alpha Generation
    quantum_gap_regime_breakout = (
        data['gap_regime_adaptive_core'] * 
        data['gap_quantum_efficiency_core'] * 
        data['gap_quantum_entropy_pressure']
    )
    
    gap_fractal_quantum_efficiency = (
        data['gap_quantum_efficiency_core'] * 
        data['gap_quantum_efficiency_ratio'] * 
        (data['amount'] / data['amount'].rolling(5).mean())
    )
    
    gap_fracture_consistency_signal = (
        abs(data['close_to_open']) * np.sign(data['close_to_open']) * 
        (data['volume'] > data['volume'].rolling(5).mean() * 1.2) * 
        data['fractal_gap_fill_ratio']
    )
    
    # Final composite alpha
    alpha = (
        quantum_gap_regime_breakout * 
        gap_fractal_quantum_efficiency * 
        gap_fracture_consistency_signal * 
        data['gap_momentum_dispersion_divergence']
    )
    
    return alpha
