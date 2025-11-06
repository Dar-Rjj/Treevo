import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Fractal Framework
    # Price Momentum Spectrum
    data['ultra_short_momentum'] = (data['close'] / data['close'].shift(2) - 1) - (data['close'] / data['close'].shift(4) - 1)
    data['short_term_momentum'] = (data['close'] / data['close'].shift(4) - 1) - (data['close'] / data['close'].shift(9) - 1)
    data['momentum_coherence'] = np.sign(data['ultra_short_momentum']) * np.sign(data['short_term_momentum'])
    
    # Fractal Range Dynamics
    data['intraday_range'] = data['high'] - data['low']
    data['multi_scale_true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(4)),
            np.abs(data['low'] - data['close'].shift(4))
        )
    )
    data['range_compression_ratio'] = data['intraday_range'] / data['intraday_range'].shift(1)
    data['range_compression_ratio'] = data['range_compression_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Volatility Spectrum
    data['close_returns'] = data['close'].pct_change()
    data['short_term_volatility'] = data['close_returns'].rolling(window=3).std()
    data['medium_term_volatility'] = data['close_returns'].rolling(window=10).std()
    data['fractal_volatility_momentum'] = ((data['high'] - data['low']) / data['close']) - ((data['high'].shift(5) - data['low'].shift(5)) / data['close'].shift(5))
    
    # Gap Momentum Acceleration System
    # Core Gap Components
    data['gap_strength'] = (np.abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)) * np.sign(data['open'] - data['close'].shift(1))
    data['gap_sustainability'] = (data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    data['gap_sustainability'] = data['gap_sustainability'].replace([np.inf, -np.inf], np.nan)
    data['opening_power'] = data['gap_strength'] * data['gap_sustainability']
    
    # Fractal Gap Enhancement
    data['fractal_gap_pressure'] = ((data['open'] / data['close'].shift(1) - 1)) * data['range_compression_ratio']
    data['gap_acceleration'] = (data['open'] - data['close'].shift(1)) * (data['ultra_short_momentum'] + data['short_term_momentum']) / 2
    data['multi_scale_gap_momentum'] = data['gap_acceleration'] * data['fractal_gap_pressure'] * data['momentum_coherence']
    
    # Pressure-Efficiency Framework
    # Microstructure Pressure Components
    data['depth_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['depth_pressure'] = data['depth_pressure'].replace([np.inf, -np.inf], np.nan)
    data['upward_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['downward_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    data['net_pressure_asymmetry'] = data['upward_pressure'] - data['downward_pressure']
    
    # Volume Dynamics
    data['volume_concentration'] = data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3))
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Calculate volume pressure asymmetry
    def calculate_volume_pressure_asymmetry(df_window):
        up_volume = df_window[df_window['close'] > df_window['open']]['volume'].sum()
        down_volume = df_window[df_window['close'] < df_window['open']]['volume'].sum()
        total_volume = df_window['volume'].sum()
        if total_volume > 0:
            return (up_volume - down_volume) / total_volume
        return 0
    
    # Rolling calculation for volume pressure asymmetry
    volume_pressure_values = []
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1].copy()
            value = calculate_volume_pressure_asymmetry(window_data)
        else:
            value = np.nan
        volume_pressure_values.append(value)
    
    data['volume_pressure_asymmetry'] = volume_pressure_values
    
    # Efficiency Metrics
    data['price_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['price_efficiency'] = data['price_efficiency'].replace([np.inf, -np.inf], np.nan)
    data['execution_efficiency'] = np.abs(data['close'] - data['open']) / data['multi_scale_true_range']
    data['execution_efficiency'] = data['execution_efficiency'].replace([np.inf, -np.inf], np.nan)
    data['pressure_volume_efficiency'] = data['net_pressure_asymmetry'] * data['volume_momentum'] * data['execution_efficiency']
    
    # Breakout and Compression Analysis
    # Breakout Detection System
    data['recent_high'] = data['high'].rolling(window=5).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    data['breakout_signal'] = (data['high'] > data['recent_high']).astype(float)
    data['breakout_strength'] = (data['high'] - data['recent_high']) / data['recent_high']
    data['breakout_strength'] = data['breakout_strength'].replace([np.inf, -np.inf], np.nan)
    
    # Compression Dynamics
    data['high_low_compression'] = data['intraday_range'] / data['intraday_range'].shift(1)
    data['high_low_compression'] = data['high_low_compression'].replace([np.inf, -np.inf], np.nan)
    data['compression_breakout'] = data['breakout_strength'] * (1 - data['high_low_compression']) * data['volume_concentration']
    data['range_pressure_interaction'] = data['net_pressure_asymmetry'] / data['intraday_range']
    data['range_pressure_interaction'] = data['range_pressure_interaction'].replace([np.inf, -np.inf], np.nan)
    
    # Multi-Scale Signal Integration
    # Core Momentum-Pressure Synthesis
    data['fractal_gap_momentum'] = data['multi_scale_gap_momentum'] * data['depth_pressure']
    data['pressure_efficiency_momentum'] = data['fractal_gap_momentum'] * data['pressure_volume_efficiency']
    data['breakout_enhancement'] = data['pressure_efficiency_momentum'] * (1 + data['breakout_signal'])
    
    # Volatility-Adaptive Components
    data['volatility_scaled_momentum'] = data['breakout_enhancement'] / data['short_term_volatility']
    data['volatility_scaled_momentum'] = data['volatility_scaled_momentum'].replace([np.inf, -np.inf], np.nan)
    data['fractal_volatility_adjustment'] = data['volatility_scaled_momentum'] * data['fractal_volatility_momentum']
    data['volume_fractal_confirmation'] = data['fractal_volatility_adjustment'] * data['volume_pressure_asymmetry'] * data['volume_momentum']
    
    # Confirmation and Alignment System
    # Momentum Confirmation Layer
    data['opening_momentum'] = data['opening_power'] * data['gap_acceleration']
    data['range_momentum'] = (data['close'] - data['close'].shift(2)) / (data['high'].shift(2) - data['low'].shift(2))
    data['range_momentum'] = data['range_momentum'].replace([np.inf, -np.inf], np.nan)
    data['momentum_decay'] = data['ultra_short_momentum'] - data['short_term_momentum']
    
    # Volume-Pressure Alignment
    data['volume_efficiency'] = data['price_efficiency'] * data['volume_concentration']
    data['compression_alignment'] = data['compression_breakout'] * data['range_compression_ratio']
    
    # Composite Alpha Construction
    # Multi-Scale Factor Integration
    data['primary_signal'] = data['volume_fractal_confirmation'] * data['opening_momentum']
    data['efficiency_boost'] = data['primary_signal'] * data['pressure_volume_efficiency']
    data['breakout_confirmation'] = data['efficiency_boost'] * data['compression_breakout']
    
    # Final Alpha Factor
    data['fractal_enhanced_output'] = data['breakout_confirmation'] * data['fractal_volatility_momentum']
    data['pressure_weighted_alpha'] = data['fractal_enhanced_output'] * (1 + np.abs(data['volume_pressure_asymmetry']))
    
    # Return the final alpha factor
    return data['pressure_weighted_alpha']
