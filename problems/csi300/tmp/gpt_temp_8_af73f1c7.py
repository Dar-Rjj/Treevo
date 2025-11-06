import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Directional Volatility Asymmetry Framework
    upside_vol_pressure = (data['high'] - data['close']) / np.maximum(data['close'] - data['low'], 1e-8)
    downside_vol_compression = (data['close'] - data['low']) / np.maximum(data['high'] - data['close'], 1e-8)
    volatility_skew_momentum = upside_vol_pressure - downside_vol_compression
    
    # Bidirectional Volume Flow System
    positive_condition = data['close'] > data['open']
    negative_condition = data['close'] < data['open']
    
    # Calculate rolling sums with proper windowing
    positive_volume_sum = data['volume'].rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(x[positive_condition.iloc[-len(x):].values]), raw=False
    )
    total_volume_3d = data['volume'].rolling(window=3, min_periods=1).sum()
    positive_flow_intensity = positive_volume_sum / np.maximum(total_volume_3d, 1e-8)
    
    negative_volume_sum = data['volume'].rolling(window=6, min_periods=1).apply(
        lambda x: np.sum(x[negative_condition.iloc[-len(x):].values]), raw=False
    )
    total_volume_6d = data['volume'].rolling(window=6, min_periods=1).sum()
    negative_flow_persistence = negative_volume_sum / np.maximum(total_volume_6d, 1e-8)
    
    flow_direction_divergence = positive_flow_intensity - negative_flow_persistence
    
    # Price Elasticity Framework
    opening_elasticity = (data['close'] - data['open']) / np.maximum(data['open'] - data['close'].shift(1), 1e-8)
    closing_elasticity = (data['close'] - data['open']) / np.maximum(data['close'].shift(1) - data['open'], 1e-8)
    elasticity_convergence = opening_elasticity * closing_elasticity
    
    # Momentum Fracture Detection
    fracture_strength = np.abs(data['close'] / data['close'].shift(1) - 1) / np.maximum(data['high'] / data['low'] - 1, 1e-8)
    fracture_recovery = ((data['close'] - data['low']) / np.maximum(data['high'] - data['low'], 1e-8)) * (data['close'] / data['open'] - 1)
    fracture_momentum = fracture_strength * fracture_recovery
    
    # Volume-Pressure Integration
    volume_pressure = (data['amount'] / np.maximum(data['volume'], 1e-8)) / np.maximum(data['amount'].shift(1) / np.maximum(data['volume'].shift(1), 1e-8), 1e-8)
    price_pressure = (data['high'] - data['low']) / np.maximum(data['close'] - data['open'], 1e-8)
    pressure_convergence = volume_pressure * price_pressure
    
    # Regime-Sensitive Weighting
    high_volatility_regime = (data['high'] - data['low']) / np.maximum(data['close'].shift(1), 1e-8)
    low_volatility_regime = (data['close'] - data['open']) / np.maximum(data['high'] - data['low'], 1e-8)
    regime_switch_signal = high_volatility_regime / np.maximum(low_volatility_regime, 1e-8)
    
    # Composite Alpha Synthesis
    core_signal = volatility_skew_momentum * flow_direction_divergence
    enhanced_signal = core_signal * elasticity_convergence * fracture_momentum
    pressure_adapted_signal = enhanced_signal * pressure_convergence
    final_alpha = pressure_adapted_signal * regime_switch_signal
    
    # Handle any infinite or NaN values
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan)
    final_alpha = final_alpha.fillna(0)
    
    return final_alpha
