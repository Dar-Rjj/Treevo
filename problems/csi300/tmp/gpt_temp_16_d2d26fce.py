import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize epsilon for numerical stability
    epsilon = 1e-8
    
    # Multi-Scale Price Efficiency
    # Micro Price Efficiency
    denominator_micro = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    price_efficiency_micro = (data['close'] - data['open']) / (denominator_micro + epsilon)
    
    # Meso Price Efficiency
    high_3d = data['high'].rolling(window=4, min_periods=4).max()
    low_3d = data['low'].rolling(window=4, min_periods=4).min()
    price_efficiency_meso = (data['close'] - data['close'].shift(3)) / (high_3d - low_3d + epsilon)
    
    # Macro Price Efficiency
    high_8d = data['high'].rolling(window=9, min_periods=9).max()
    low_8d = data['low'].rolling(window=9, min_periods=9).min()
    price_efficiency_macro = (data['close'] - data['close'].shift(8)) / (high_8d - low_8d + epsilon)
    
    # Multi-Scale Volume Efficiency
    # Micro Volume Efficiency
    volume_efficiency_micro = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Meso Volume Efficiency
    volume_sum_3d = data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)
    volume_efficiency_meso = data['volume'] / (volume_sum_3d + epsilon)
    
    # Macro Volume Efficiency (Volume Distribution Asymmetry)
    volume_dist_asymmetry = (
        data['volume'] * (data['open'] - data['low']) / (data['high'] - data['low'] + epsilon) - 
        data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'] + epsilon)
    )
    
    # Asymmetric Efficiency Dynamics
    # Efficiency Asymmetry
    up_efficiency = price_efficiency_micro.where(data['close'] > data['close'].shift(1), 0)
    down_efficiency = price_efficiency_micro.where(data['close'] < data['close'].shift(1), 0)
    efficiency_asymmetry = up_efficiency - down_efficiency
    
    # Volume Concentration
    volume_spike = data['volume'] / (data['volume'].shift(1) + epsilon) - 1
    
    # Volume Persistence (count consecutive days with increasing volume)
    volume_increase = (data['volume'] > data['volume'].shift(1)).astype(int)
    volume_persistence = volume_increase.rolling(window=5, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == 1 and x.iloc[i-1] == 1]), 
        raw=False
    )
    
    # Alignment Consistency
    volume_sign_consistency = volume_efficiency_micro.rolling(window=6, min_periods=6).apply(
        lambda x: sum(x.iloc[i] * x.iloc[i-1] > 0 for i in range(1, len(x))), 
        raw=False
    )
    
    # Divergence Framework
    # Efficiency Divergence
    price_efficiency_divergence = price_efficiency_micro * price_efficiency_meso * price_efficiency_macro
    volume_efficiency_divergence = volume_efficiency_micro * volume_efficiency_meso * volume_dist_asymmetry
    cross_divergence = price_efficiency_divergence * volume_efficiency_divergence
    
    # Range Efficiency
    range_efficiency_micro = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + epsilon)
    range_efficiency_meso = (data['close'] - data['close'].shift(3)) / (high_3d - low_3d + epsilon)
    range_efficiency_macro = (data['close'] - data['close'].shift(8)) / (high_8d - low_8d + epsilon)
    
    # Pressure Analysis
    # Session Pressure
    upper_pressure = (data['high'] - data['close']) / (data['high'] - data['low'] + epsilon)
    lower_pressure = (data['close'] - data['low']) / (data['high'] - data['low'] + epsilon)
    session_imbalance = upper_pressure - lower_pressure
    
    # Breakout Pressure
    high_prev_3d = data['high'].shift(1).rolling(window=3, min_periods=3).max()
    low_prev_3d = data['low'].shift(1).rolling(window=3, min_periods=3).min()
    resistance = (data['high'] - high_prev_3d) / (data['close'].shift(1) + epsilon)
    support = (low_prev_3d - data['low']) / (data['close'].shift(1) + epsilon)
    breakout_asymmetry = resistance - support
    
    # Regime Classification
    # Efficiency Regime
    high_efficiency = (price_efficiency_micro > 0) & (price_efficiency_meso > 0)
    low_efficiency = (price_efficiency_micro < 0) & (price_efficiency_meso < 0)
    mixed_efficiency = (price_efficiency_micro * price_efficiency_meso < 0)
    
    # Volume Regime
    high_volume = volume_efficiency_meso > 0.4
    low_volume = volume_efficiency_meso < 0.2
    normal_volume = (~high_volume) & (~low_volume)
    
    # Volatility Regime
    vol_recent = (data['high'].rolling(window=3, min_periods=3).max() - 
                 data['low'].rolling(window=3, min_periods=3).min())
    vol_prev = (data['high'].shift(3).rolling(window=3, min_periods=3).max() - 
               data['low'].shift(3).rolling(window=3, min_periods=3).min())
    vol_ratio = vol_recent / (vol_prev + epsilon)
    
    expanding_vol = vol_ratio > 1
    contracting_vol = vol_ratio < 1
    stable_vol = (vol_ratio >= 0.9) & (vol_ratio <= 1.1)
    
    # Quality Enhancement
    # Efficiency Consistency
    efficiency_consistency = price_efficiency_micro.rolling(window=6, min_periods=6).apply(
        lambda x: sum(x.iloc[i] * x.iloc[i-1] > 0 for i in range(1, len(x))), 
        raw=False
    )
    
    # Volume Flow Consistency
    volume_flow_consistency = volume_dist_asymmetry.rolling(window=6, min_periods=6).apply(
        lambda x: sum(x.iloc[i] * x.iloc[i-1] > 0 for i in range(1, len(x))), 
        raw=False
    )
    
    # Alpha Synthesis
    # High Efficiency Expanding Volume Factor
    high_eff_expanding_factor = (
        efficiency_asymmetry * 
        volume_efficiency_meso * 
        cross_divergence * 
        breakout_asymmetry
    )
    
    # Mixed Efficiency Breakout Factor
    mixed_eff_breakout_factor = (
        cross_divergence * 
        breakout_asymmetry * 
        volume_efficiency_micro
    )
    
    # Low Volume Efficiency Reversal Factor
    low_vol_reversal_factor = (
        efficiency_asymmetry * 
        volume_persistence * 
        session_imbalance
    )
    
    # Strong Divergence Normal Volume Factor
    strong_divergence_factor = (
        cross_divergence * 
        range_efficiency_micro * 
        range_efficiency_meso * 
        range_efficiency_macro
    )
    
    # Composite Alpha
    composite_alpha = pd.Series(index=data.index, dtype=float)
    
    # Regime-based alpha assignment
    high_eff_expanding_mask = high_efficiency & expanding_vol & high_volume
    mixed_eff_breakout_mask = mixed_efficiency & (abs(breakout_asymmetry) > 0.02)
    low_vol_reversal_mask = low_volume & mixed_efficiency
    strong_divergence_mask = (cross_divergence > 0) & normal_volume & stable_vol
    
    composite_alpha[high_eff_expanding_mask] = high_eff_expanding_factor[high_eff_expanding_mask]
    composite_alpha[mixed_eff_breakout_mask] = mixed_eff_breakout_factor[mixed_eff_breakout_mask]
    composite_alpha[low_vol_reversal_mask] = low_vol_reversal_factor[low_vol_reversal_mask]
    composite_alpha[strong_divergence_mask] = strong_divergence_factor[strong_divergence_mask]
    
    # Default case
    default_mask = ~(high_eff_expanding_mask | mixed_eff_breakout_mask | 
                    low_vol_reversal_mask | strong_divergence_mask)
    composite_alpha[default_mask] = (
        efficiency_consistency[default_mask] * 
        volume_flow_consistency[default_mask] * 
        cross_divergence[default_mask] * 
        efficiency_asymmetry[default_mask]
    )
    
    return composite_alpha
