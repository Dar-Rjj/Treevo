import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Gap Analysis
    # Gap Momentum Framework
    data['micro_gap'] = data['open'] / data['close'].shift(1) - 1
    data['meso_gap'] = data['open'] / data['close'].shift(3) - 1
    data['macro_gap'] = data['open'] / data['close'].shift(8) - 1
    data['gap_fractal_ratio'] = (data['micro_gap'] * data['macro_gap']) / (abs(data['meso_gap']) + 1e-8)
    
    # Gap Persistence Analysis
    data['micro_persistence'] = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['open'].shift(1) - data['close'].shift(2))
    data['meso_persistence'] = np.sign(data['open'] - data['close'].shift(3)) * np.sign(data['open'].shift(3) - data['close'].shift(6))
    data['macro_persistence'] = np.sign(data['open'] - data['close'].shift(8)) * np.sign(data['open'].shift(8) - data['close'].shift(16))
    data['multi_scale_gap_persistence'] = data['micro_persistence'] + data['meso_persistence'] + data['macro_persistence']
    
    # Gap Acceleration Analysis
    data['micro_acceleration'] = (data['open'] - data['close'].shift(1)) - (data['open'].shift(1) - data['close'].shift(2))
    data['meso_acceleration'] = (data['open'] - data['close'].shift(3)) - (data['open'].shift(3) - data['close'].shift(6))
    data['macro_acceleration'] = (data['open'] - data['close'].shift(8)) - (data['open'].shift(8) - data['close'].shift(16))
    data['multi_scale_gap_acceleration'] = data['micro_acceleration'] + data['meso_acceleration'] + data['macro_acceleration']
    
    # Volume-Price Anchoring System
    # Multi-Scale Volume Momentum
    data['volume_micro_change'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_meso_change'] = data['volume'] / data['volume'].shift(4) - 1
    data['volume_macro_change'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_fractal_dimension'] = (data['volume_micro_change'] * data['volume_macro_change']) / (abs(data['volume_meso_change']) + 1e-8)
    
    # Volume-Pressure Anchoring
    data['micro_pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    
    # Rolling calculations for meso and macro pressure
    data['meso_pressure'] = (
        (data['close'].rolling(window=3).sum() - data['low'].rolling(window=3).sum()) / 
        (data['high'].rolling(window=3).sum() - data['low'].rolling(window=3).sum() + 1e-8)
    ) * data['volume'].rolling(window=3).sum()
    
    data['macro_pressure'] = (
        (data['close'].rolling(window=6).sum() - data['low'].rolling(window=6).sum()) / 
        (data['high'].rolling(window=6).sum() - data['low'].rolling(window=6).sum() + 1e-8)
    ) * data['volume'].rolling(window=6).sum()
    
    data['multi_scale_volume_pressure'] = (data['micro_pressure'] * data['macro_pressure']) / (abs(data['meso_pressure']) + 1e-8)
    
    # Volume Trend Anchoring
    def count_volume_increase(window_data):
        return sum(window_data.iloc[i] > window_data.iloc[i-1] for i in range(1, len(window_data)))
    
    data['micro_trend'] = data['volume'].rolling(window=3).apply(count_volume_increase, raw=False)
    data['meso_trend'] = data['volume'].rolling(window=6).apply(count_volume_increase, raw=False)
    data['macro_trend'] = data['volume'].rolling(window=11).apply(count_volume_increase, raw=False)
    data['multi_scale_volume_trend'] = data['micro_trend'] + data['meso_trend'] + data['macro_trend']
    
    # Multi-Scale Volatility Regime Detection
    # Volatility Fractal Structure
    data['micro_volatility'] = data['high'] - data['low']
    data['meso_volatility'] = data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()
    data['macro_volatility'] = data['high'].rolling(window=6).max() - data['low'].rolling(window=6).min()
    data['volatility_fractal_ratio'] = (data['micro_volatility'] * data['macro_volatility']) / (data['meso_volatility'] + 1e-8)
    
    # Multi-Scale Temporal Alignment
    # Gap-Momentum Alignment
    data['micro_alignment'] = np.sign(data['micro_gap']) * np.sign(data['volume_micro_change'])
    data['meso_alignment'] = np.sign(data['meso_gap']) * np.sign(data['volume_meso_change'])
    data['macro_alignment'] = np.sign(data['macro_gap']) * np.sign(data['volume_macro_change'])
    data['gap_volume_alignment_score'] = data['micro_alignment'] + data['meso_alignment'] + data['macro_alignment']
    
    # Persistence-Pressure Alignment
    data['micro_persistence_pressure'] = np.sign(data['micro_persistence']) * np.sign(data['micro_pressure'] - data['meso_pressure'])
    data['meso_persistence_pressure'] = np.sign(data['meso_persistence']) * np.sign(data['meso_pressure'] - data['macro_pressure'])
    data['macro_persistence_pressure'] = np.sign(data['macro_persistence']) * np.sign(data['macro_pressure'] - data['micro_pressure'])
    data['persistence_pressure_alignment_score'] = data['micro_persistence_pressure'] + data['meso_persistence_pressure'] + data['macro_persistence_pressure']
    
    data['temporal_alignment_multiplier'] = data['gap_volume_alignment_score'] * data['persistence_pressure_alignment_score']
    
    # Composite Alpha Generation
    # Core components
    data['core_gap_signal'] = data['gap_fractal_ratio'] * data['multi_scale_gap_persistence']
    data['volume_anchor'] = data['volume_fractal_dimension'] * data['multi_scale_volume_pressure']
    data['trend_signal'] = data['multi_scale_gap_acceleration'] * data['multi_scale_volume_trend']
    
    # Final alpha factor
    data['alpha_factor'] = (
        data['core_gap_signal'] * 
        data['volume_anchor'] * 
        data['trend_signal'] * 
        data['volatility_fractal_ratio'] * 
        data['temporal_alignment_multiplier']
    )
    
    return data['alpha_factor']
