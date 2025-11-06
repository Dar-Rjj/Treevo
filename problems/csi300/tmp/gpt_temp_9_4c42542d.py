import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Volatility Convergence Analysis
    # Short-Term Amplitude Pressure (3-day)
    df['upside_pressure_3d'] = 0.0
    df['downside_pressure_3d'] = 0.0
    
    for i in range(3):
        df['upside_pressure_3d'] += (df['high'].shift(i) - df['close'].shift(i)) * df['volume'].shift(i)
        df['downside_pressure_3d'] += (df['close'].shift(i) - df['low'].shift(i)) * df['volume'].shift(i)
    
    # Avoid division by zero and negative values
    df['short_term_asymmetry'] = np.log(
        np.maximum(df['upside_pressure_3d'], 1e-6) / np.maximum(df['downside_pressure_3d'], 1e-6)
    )
    
    # Medium-Term Amplitude Pressure (8-day)
    df['upside_pressure_8d'] = 0.0
    df['downside_pressure_8d'] = 0.0
    
    for i in range(8):
        df['upside_pressure_8d'] += (df['high'].shift(i) - df['close'].shift(i)) * df['volume'].shift(i)
        df['downside_pressure_8d'] += (df['close'].shift(i) - df['low'].shift(i)) * df['volume'].shift(i)
    
    df['medium_term_asymmetry'] = np.log(
        np.maximum(df['upside_pressure_8d'], 1e-6) / np.maximum(df['downside_pressure_8d'], 1e-6)
    )
    
    # Amplitude Convergence Detection
    df['amplitude_alignment'] = np.sign(df['short_term_asymmetry']) * np.sign(df['medium_term_asymmetry'])
    df['convergence_strength'] = df['short_term_asymmetry'] * df['medium_term_asymmetry']
    
    # Price-Momentum Acceleration Component
    df['short_term_momentum'] = df['close'] - df['close'].shift(2)
    df['medium_term_momentum'] = df['close'] - df['close'].shift(5)
    df['momentum_acceleration'] = df['short_term_momentum'] - df['medium_term_momentum']
    
    # Volume-Efficiency Integration
    df['price_efficiency_ratio'] = (df['close'] - df['open']) / np.maximum(df['high'] - df['low'], 1e-6)
    df['volume_intensity'] = df['volume'] / np.maximum(df['high'] - df['low'], 1e-6)
    
    # Volume-Regime Transition Analysis
    df['amplitude_ratio'] = df['short_term_asymmetry'] / np.maximum(np.abs(df['medium_term_asymmetry']), 1e-6)
    df['transition_flag'] = (df['amplitude_ratio'].shift(1) < 1.0) & (df['amplitude_ratio'] >= 1.0)
    
    # Calculate volume efficiency persistence
    df['efficiency_sign'] = np.sign(df['price_efficiency_ratio'])
    df['efficiency_persistence'] = 0
    for i in range(1, len(df)):
        if df['efficiency_sign'].iloc[i] == df['efficiency_sign'].iloc[i-1]:
            df.loc[df.index[i], 'efficiency_persistence'] = df['efficiency_persistence'].iloc[i-1] + 1
        else:
            df.loc[df.index[i], 'efficiency_persistence'] = 1
    
    # Signal Integration and Filtering
    df['raw_signal'] = df['convergence_strength'] * df['momentum_acceleration']
    
    # Apply Efficiency Filtering
    df['efficiency_filtered'] = df['raw_signal'] * df['price_efficiency_ratio'] * df['volume_intensity']
    
    # Regime-Transition Enhancement
    df['transition_multiplier'] = 1.0
    df.loc[df['transition_flag'], 'transition_multiplier'] = 1.5
    df['regime_enhanced'] = df['efficiency_filtered'] * df['transition_multiplier'] * (1 + df['efficiency_persistence'] * 0.1)
    
    # Volatility Context and Range Analysis
    df['recent_amplitude'] = (
        (df['high'] - df['low']) + 
        (df['high'].shift(1) - df['low'].shift(1)) +
        (df['high'].shift(2) - df['low'].shift(2)) +
        (df['high'].shift(3) - df['low'].shift(3)) +
        (df['high'].shift(4) - df['low'].shift(4))
    ) / 5.0
    
    # Range Breakout Confirmation
    df['range_15d_high'] = df['high'].rolling(window=15, min_periods=1).max()
    df['range_15d_low'] = df['low'].rolling(window=15, min_periods=1).min()
    df['breakout_multiplier'] = 1.0
    df.loc[df['close'] > df['range_15d_high'].shift(1), 'breakout_multiplier'] = 1.3
    df.loc[df['close'] < df['range_15d_low'].shift(1), 'breakout_multiplier'] = 1.3
    
    # Contextual Signal Scaling
    df['contextual_signal'] = df['regime_enhanced'] * df['recent_amplitude'] * df['breakout_multiplier']
    
    # Generate Final Alpha Factor with Cube Root Transformation
    df['final_factor'] = np.sign(df['contextual_signal']) * np.power(np.abs(df['contextual_signal']), 1/3)
    
    # Clean up intermediate columns
    result = df['final_factor'].copy()
    
    return result
