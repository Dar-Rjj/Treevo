import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Quantum Fracture-Gap Components
    data['Fracture_Gap_Wave'] = (data['close'] - data['close'].shift(1)) * abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 0.001)
    data['Fracture_Gap_Particle'] = (data['high'] - data['low']) * (data['open'] - data['close'].shift(1)) / (abs(data['close'] - data['open']) + 0.001)
    data['Quantum_Fracture_Gap_State'] = data['Fracture_Gap_Wave'] * data['Fracture_Gap_Particle']
    
    # Fracture-Gap Pressure Framework
    data['Fracture_Gap_Volume_Certainty'] = data['volume'] / (abs(data['open'] - data['close'].shift(1)) + 0.001)
    data['Fracture_Gap_Pressure_Uncertainty'] = abs(data['close'] - data['open']) / (abs(data['open'] - data['close'].shift(1)) + 0.001)
    data['Fracture_Gap_Pressure_Product'] = data['Fracture_Gap_Volume_Certainty'] * data['Fracture_Gap_Pressure_Uncertainty']
    
    # Quantum Fracture-Gap Synthesis
    data['Wave_Particle_Fracture_Coupling'] = data['Quantum_Fracture_Gap_State'] * data['Fracture_Gap_Pressure_Product']
    data['Volume_Fracture_Gap_Correlation'] = data['volume'] * data['Fracture_Gap_Particle'] / (abs(data['open'] - data['close'].shift(1)) + 0.001)
    data['Quantum_Fracture_Gap_Alpha'] = data['Wave_Particle_Fracture_Coupling'] * data['Volume_Fracture_Gap_Correlation']
    
    # Multi-Scale Fracture Dynamics
    data['Fast_Fracture'] = abs(data['close'] - data['open']) * (data['volume'] / (data['volume'].shift(1) + 0.001))
    data['Medium_Fracture'] = abs(data['close'] - data['open'].shift(3)) * (data['volume'] / (data['volume'].shift(3) + 0.001))
    data['Slow_Fracture'] = abs(data['close'] - data['open'].shift(5)) * (data['volume'] / (data['volume'].shift(5) + 0.001))
    
    # Fracture Phase Analysis
    data['Fast_Medium_Fracture_Phase'] = data['Fast_Fracture'] - data['Medium_Fracture']
    data['Medium_Slow_Fracture_Phase'] = data['Medium_Fracture'] - data['Slow_Fracture']
    data['Fracture_Phase_Coherence'] = data['Fast_Medium_Fracture_Phase'] * data['Medium_Slow_Fracture_Phase']
    
    # Spectral Fracture Integration
    data['Fast_Medium_Fracture_Divergence'] = data['Fast_Fracture'] / (data['Medium_Fracture'] + 0.001)
    data['Medium_Slow_Fracture_Convergence'] = data['Medium_Fracture'] / (data['Slow_Fracture'] + 0.001)
    data['Spectral_Fracture_Alpha'] = data['Fracture_Phase_Coherence'] * data['Fast_Medium_Fracture_Divergence'] * data['Medium_Slow_Fracture_Convergence']
    
    # Fracture Cluster Dynamics
    def calc_fracture_volume_fractal(window_data):
        if len(window_data) < 2:
            return np.nan
        long_window = window_data[-7:] if len(window_data) >= 7 else window_data
        short_window = window_data[-2:] if len(window_data) >= 2 else window_data
        
        long_range = np.max(long_window) - np.min(long_window) if len(long_window) > 0 else 1
        short_range = np.max(short_window) - np.min(short_window) if len(short_window) > 0 else 1
        
        return np.log(long_range + 0.001) / np.log(short_range + 0.001)
    
    # Calculate fracture volume products for rolling windows
    data['fracture_volume_product'] = abs(data['open'] - data['close'].shift(1)) * data['volume']
    data['fracture_volume_fractal'] = data['fracture_volume_product'].rolling(window=7, min_periods=2).apply(calc_fracture_volume_fractal, raw=False)
    
    # Fracture Turnover Momentum
    data['volume_fracture_product'] = data['volume'] * abs(data['close'] - data['open'])
    data['max_prev_volume_fracture'] = data['volume_fracture_product'].shift(1).rolling(window=4, min_periods=1).max()
    data['Fracture_Turnover_Momentum'] = data['volume_fracture_product'] / (data['max_prev_volume_fracture'] + 0.001)
    
    # Fracture Cluster Duration (simplified implementation)
    data['fracture_threshold'] = data['fracture_volume_product'].rolling(window=7, min_periods=1).median() * 2.5
    data['above_threshold'] = (data['fracture_volume_product'] > data['fracture_threshold']).astype(int)
    data['Fracture_Cluster_Duration'] = data['above_threshold'].rolling(window=7, min_periods=1).sum()
    
    # Fracture Flow Asymmetry
    data['Fracture_Volume_Momentum'] = (data['volume'] / (data['volume'].shift(1) + 0.001)) - (data['volume'].shift(1) / (data['volume'].shift(2) + 0.001))
    data['Fracture_Amount_Efficiency'] = (data['amount'] / (data['volume'] + 0.001)) / (data['amount'].shift(1) / (data['volume'].shift(1) + 0.001))
    
    # Upside Fracture Ratio
    data['upside_day'] = (data['close'] > data['open']).astype(int)
    data['upside_volume'] = data['upside_day'] * data['volume']
    data['Upside_Fracture_Ratio'] = data['upside_volume'].rolling(window=10, min_periods=1).sum() / (data['volume'].rolling(window=10, min_periods=1).sum() + 0.001)
    
    data['Fracture_Asymmetry'] = data['Fracture_Volume_Momentum'] * data['Fracture_Amount_Efficiency'] * data['Upside_Fracture_Ratio']
    
    # Fracture Confirmation Alpha
    data['Fracture_Confirmation_Alpha'] = (data['fracture_volume_fractal'] * data['Fracture_Turnover_Momentum']) * data['Fracture_Asymmetry']
    
    # Volume-Fracture Enhancement
    data['volume_sum_3d'] = data['volume'].rolling(window=3, min_periods=1).sum()
    data['Volume_Fracture_Enhancement'] = (data['volume'] / (data['volume_sum_3d'] + 0.001)) * (abs(data['open'] - data['close'].shift(1)) / (data['open'] + data['close'] + 0.001))
    
    # Quantum-Fracture Integration
    data['Quantum_Fracture_Fusion'] = data['Quantum_Fracture_Gap_Alpha'] * data['Spectral_Fracture_Alpha']
    data['Spectral_Fracture_Coupling'] = data['Fracture_Phase_Coherence'] * data['Fracture_Confirmation_Alpha']
    data['Base_Quantum_Fracture_Alpha'] = data['Quantum_Fracture_Fusion'] * data['Spectral_Fracture_Coupling'] * data['Volume_Fracture_Enhancement']
    
    # Quantum-Fracture State Assessment
    data['Fracture_Microstructure_Strength'] = abs(data['Quantum_Fracture_Gap_State']) + abs(data['Fracture_Gap_Pressure_Product'])
    data['Spectral_Fracture_Strength'] = abs(data['Fast_Fracture']) + abs(data['Fracture_Phase_Coherence'])
    data['Fracture_Confirmation_Strength'] = abs(data['fracture_volume_fractal']) + abs(data['Fracture_Asymmetry'])
    data['Volume_Strength'] = abs(data['volume'] / (data['volume_sum_3d'] + 0.001)) + abs(data['Fracture_Gap_Volume_Certainty'])
    
    # Quantum-Fracture Weight Synthesis
    total_strength = (data['Fracture_Microstructure_Strength'] + data['Spectral_Fracture_Strength'] + 
                     data['Fracture_Confirmation_Strength'] + data['Volume_Strength'] + 0.001)
    
    data['Fracture_Microstructure_Weight'] = data['Fracture_Microstructure_Strength'] / total_strength
    data['Spectral_Fracture_Weight'] = data['Spectral_Fracture_Strength'] / total_strength
    data['Fracture_Confirmation_Weight'] = data['Fracture_Confirmation_Strength'] / total_strength
    data['Volume_Weight'] = data['Volume_Strength'] / total_strength
    
    data['Weighted_Quantum_Fracture_Alpha'] = (
        data['Quantum_Fracture_Gap_Alpha'] * data['Fracture_Microstructure_Weight'] +
        data['Spectral_Fracture_Alpha'] * data['Spectral_Fracture_Weight'] +
        data['Fracture_Confirmation_Alpha'] * data['Fracture_Confirmation_Weight'] +
        (data['volume'] / (data['volume_sum_3d'] + 0.001)) * data['Volume_Weight']
    )
    
    # Dynamic Fracture Enhancement
    data['Fracture_Compression'] = abs(data['open'] - data['close'].shift(1)) / (abs(data['open'].shift(1) - data['close'].shift(2)) + 0.001)
    data['Fracture_Expansion'] = (
        data['fracture_volume_product'].rolling(window=3, min_periods=1).max() - 
        data['fracture_volume_product'].rolling(window=3, min_periods=1).min()
    ) / (abs(data['open'].shift(1) - data['close'].shift(2)) + 0.001)
    
    data['Quantum_Fracture_Breakout_Multiplier'] = 1 + ((data['Fracture_Compression'] < 0.6) * data['Fracture_Expansion'])
    
    # Volume Fracture Adjustment
    data['volume_sum_7d'] = data['volume'].rolling(window=7, min_periods=1).sum()
    data['Volume_Fracture_State'] = data['Weighted_Quantum_Fracture_Alpha'] * (data['volume'] / (data['volume_sum_7d'] + 0.001))
    data['Amount_Fracture'] = data['Volume_Fracture_State'] * (data['amount'] / (data['volume'] + 0.001))
    
    data['Quantum_Fracture_Resonance_Alpha'] = data['Amount_Fracture'] * data['Quantum_Fracture_Breakout_Multiplier']
    
    # Fracture Momentum Acceleration Framework
    data['Micro_Fractal_Acceleration'] = (
        (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001) - 
        (data['close'].shift(1) - data['open'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 0.001)
    )
    
    data['Fracture_Momentum_Convergence'] = (
        np.sign(data['Fast_Fracture']) * np.sign(data['Medium_Fracture']) * np.sign(data['Slow_Fracture'])
    )
    
    data['Fracture_Acceleration_Divergence'] = (
        data['Micro_Fractal_Acceleration'] * (data['Fast_Medium_Fracture_Phase'] + data['Medium_Slow_Fracture_Phase'])
    )
    
    # Efficiency Fracture Mapping
    data['Intraday_Fracture_Efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['max_overnight_fracture'] = abs(data['open'] - data['close'].shift(1)).rolling(window=7, min_periods=1).max()
    data['Overnight_Fracture_Efficiency'] = abs(data['open'] - data['close'].shift(1)) / (data['max_overnight_fracture'] + 0.001)
    data['Fracture_Efficiency_Ratio'] = data['Intraday_Fracture_Efficiency'] / (data['Overnight_Fracture_Efficiency'] + 0.001)
    
    data['Fracture_Anchor_Persistence'] = (
        np.sign(data['close'] - ((data['high'] + data['low']) / 2)) * 
        np.sign(data['close'].shift(1) - ((data['high'].shift(1) + data['low'].shift(1)) / 2))
    )
    
    # Final Quantum Fracture-Gap Resonance Alpha
    data['Core_Quantum_Fracture_Signal'] = data['Quantum_Fracture_Resonance_Alpha'] * data['Fracture_Momentum_Convergence']
    data['Spectral_Fracture_Enhancement'] = data['Core_Quantum_Fracture_Signal'] * data['Fracture_Efficiency_Ratio']
    data['Quantum_Fracture_Momentum_Integrated'] = data['Spectral_Fracture_Enhancement'] * data['Fracture_Acceleration_Divergence']
    data['Fracture_Efficiency_Aligned'] = data['Quantum_Fracture_Momentum_Integrated'] * data['Fracture_Anchor_Persistence']
    
    # Final alpha factor
    data['Quantum_Fracture_Gap_Resonance_Alpha'] = (
        data['Fracture_Efficiency_Aligned'] * data['Fracture_Phase_Coherence'] * data['Volume_Fracture_Enhancement']
    )
    
    return data['Quantum_Fracture_Gap_Resonance_Alpha']
