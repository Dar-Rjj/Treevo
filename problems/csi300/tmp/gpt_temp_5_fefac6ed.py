import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price Fracture Asymmetry
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    up_fracture = np.where(data['close'] > data['prev_close'], 
                          (data['close'] - data['prev_close']) / (data['prev_high'] - data['prev_low'] + 1e-8), 
                          0)
    down_fracture = np.where(data['close'] < data['prev_close'], 
                            (data['prev_close'] - data['close']) / (data['prev_high'] - data['prev_low'] + 1e-8), 
                            0)
    fracture_asymmetry = up_fracture - down_fracture
    
    # Volume-Price Entropy Asymmetry
    up_volume_eff = np.where(data['close'] > data['open'], 
                            data['volume'] * (data['close'] - data['open']), 
                            0)
    down_volume_eff = np.where(data['close'] < data['open'], 
                              data['volume'] * (data['open'] - data['close']), 
                              0)
    volume_eff_asymmetry = up_volume_eff - down_volume_eff
    
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_volume2'] = data['volume'].shift(2)
    volume_entropy_asymmetry = (data['volume'] / (data['prev_volume'] + data['prev_volume2'] + 1e-8)) * volume_eff_asymmetry
    
    # Fracture-Entropy Regime Detection
    fracture_entropy = fracture_asymmetry * (data['high'] - data['low']) / (data['close'] - data['open'] + 1e-8)
    
    data['prev_amount'] = data['amount'].shift(1)
    high_asymmetry_regime = (fracture_asymmetry > volume_entropy_asymmetry).astype(float)
    low_asymmetry_regime = (fracture_asymmetry < volume_entropy_asymmetry * 0.5).astype(float)
    transition_regime = (np.abs(fracture_asymmetry - volume_entropy_asymmetry) > 
                        np.abs(data['amount'] / (data['prev_amount'] + 1e-8) - 1)).astype(float)
    
    # Multi-Scale Fractal-Asymmetry Dynamics
    # Short-Term Hurst (5-day)
    short_hurst = []
    for i in range(len(data)):
        if i >= 4:
            window = data['close'].iloc[i-4:i+1]
            range_val = window.max() - window.min()
            std_val = window.std()
            if std_val > 0:
                hurst = np.log(range_val / std_val) / np.log(5)
                short_hurst.append(hurst)
            else:
                short_hurst.append(0)
        else:
            short_hurst.append(0)
    
    # Medium-Term Hurst (20-day)
    medium_hurst = []
    for i in range(len(data)):
        if i >= 19:
            window = data['close'].iloc[i-19:i+1]
            range_val = window.max() - window.min()
            std_val = window.std()
            if std_val > 0:
                hurst = np.log(range_val / std_val) / np.log(20)
                medium_hurst.append(hurst)
            else:
                medium_hurst.append(0)
        else:
            medium_hurst.append(0)
    
    fractal_asymmetry_change = np.array(medium_hurst) - np.array(short_hurst) * np.sign(fracture_asymmetry)
    
    # Order Flow Asymmetry
    micro_imbalance_asymmetry = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * data['volume'] * np.sign(data['close'] - data['open'])
    
    volume_fractal_asymmetry = (np.log(data['volume'] / (data['prev_volume'] + 1e-8)) / 
                               np.log((data['high'] - data['low']) / (data['prev_high'] - data['prev_low'] + 1e-8) + 1e-8)) * volume_eff_asymmetry
    
    order_imbalance_asymmetry = (data['amount'] / (data['volume'] + 1e-8)) * volume_eff_asymmetry
    
    # Fracture-Fractal Synchronization
    asymmetry_flow_coupling = fracture_entropy * micro_imbalance_asymmetry
    fractal_asymmetry_alignment = fractal_asymmetry_change * volume_entropy_asymmetry
    volume_fractal_asymmetry_sync = volume_fractal_asymmetry * fracture_asymmetry
    
    # Regime-Adaptive Momentum Integration
    data['prev_close_3'] = data['close'].shift(3)
    price_momentum = data['close'] / (data['prev_close_3'] + 1e-8) - 1
    asymmetry_weighted_momentum = price_momentum * fracture_asymmetry
    entropy_aligned_momentum = asymmetry_weighted_momentum * fracture_entropy
    
    # Volume-Pressure Asymmetry
    up_tick_pressure = []
    down_tick_pressure = []
    for i in range(len(data)):
        if i >= 4:
            up_sum = 0
            down_sum = 0
            for j in range(5):
                idx = i - j
                if idx > 0 and data['close'].iloc[idx] > data['close'].iloc[idx-1]:
                    up_sum += data['volume'].iloc[idx]
                elif idx > 0 and data['close'].iloc[idx] < data['close'].iloc[idx-1]:
                    down_sum += data['volume'].iloc[idx]
            up_tick_pressure.append(up_sum)
            down_tick_pressure.append(down_sum)
        else:
            up_tick_pressure.append(0)
            down_tick_pressure.append(0)
    
    pressure_asymmetry = (np.array(up_tick_pressure) / (np.array(up_tick_pressure) + np.array(down_tick_pressure) + 1e-8)) * volume_eff_asymmetry
    
    # Regime-Dependent Momentum
    high_asymmetry_momentum = entropy_aligned_momentum * high_asymmetry_regime
    low_asymmetry_momentum = price_momentum * low_asymmetry_regime
    transition_momentum = pressure_asymmetry * transition_regime
    
    # Efficiency-Asymmetry Entropy Patterns
    daily_efficiency = np.abs(data['close'] - data['prev_close']) / (data['high'] - data['low'] + 1e-8)
    volume_efficiency = data['volume'] / (data['amount'] / (data['high'] - data['low'] + 1e-8) + 1e-8)
    asymmetric_efficiency = daily_efficiency * fracture_asymmetry
    
    price_efficiency_asymmetry = daily_efficiency * fracture_asymmetry
    volume_efficiency_entropy = volume_efficiency * volume_entropy_asymmetry
    cross_efficiency_asymmetry = np.abs(price_efficiency_asymmetry - volume_efficiency_entropy)
    
    # Regime-Adaptive Efficiency
    high_asymmetry_efficiency = asymmetric_efficiency * high_asymmetry_regime
    low_asymmetry_efficiency = volume_efficiency * low_asymmetry_regime
    transition_efficiency = cross_efficiency_asymmetry * transition_regime
    
    # Composite Signal Generation
    core_synchronization = asymmetry_flow_coupling * fractal_asymmetry_alignment
    pressure_enhanced_sync = core_synchronization * pressure_asymmetry
    efficiency_weighted_sync = pressure_enhanced_sync * asymmetric_efficiency
    
    high_regime_component = high_asymmetry_momentum * high_asymmetry_efficiency
    low_regime_component = low_asymmetry_momentum * low_asymmetry_efficiency
    transition_component = transition_momentum * transition_efficiency
    
    volume_fracture_alignment = volume_eff_asymmetry * fracture_asymmetry
    entropy_efficiency_alignment = cross_efficiency_asymmetry * volume_entropy_asymmetry
    regime_adaptive_alignment = (high_regime_component + low_regime_component + transition_component) * volume_fracture_alignment
    
    # Multi-Scale Factor Refinement
    fracture_entropy_asymmetry_core = efficiency_weighted_sync * regime_adaptive_alignment
    momentum_asymmetry_enhancement = fracture_entropy_asymmetry_core * asymmetry_weighted_momentum
    volume_pressure_confirmation = momentum_asymmetry_enhancement * pressure_asymmetry
    
    efficiency_boosted_factor = volume_pressure_confirmation * daily_efficiency
    fracture_asymmetry_weighted = efficiency_boosted_factor * fracture_asymmetry
    volume_distribution_adjusted = fracture_asymmetry_weighted / (data['amount'] / (data['volume'] + 1e-8) + 1e-8)
    
    # Regime-Stabilized Signals
    high_regime_stabilized = volume_distribution_adjusted * high_asymmetry_regime
    low_regime_stabilized = volume_distribution_adjusted * low_asymmetry_regime
    transition_stabilized = volume_distribution_adjusted * transition_regime
    
    # Final Alpha Generation
    primary_factor = high_regime_stabilized + low_regime_stabilized + transition_stabilized
    fractal_asymmetry_confirmation = primary_factor * fractal_asymmetry_change
    final_alpha = fractal_asymmetry_confirmation * volume_efficiency
    
    return pd.Series(final_alpha, index=data.index)
