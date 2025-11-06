import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Gap Efficiency Framework
    # Volatility-Weighted Gap Components
    data['prev_close'] = data['close'].shift(1)
    data['gap_magnitude'] = np.where(data['open'] != data['prev_close'], 
                                   np.abs(data['open'] - data['prev_close']), 1e-8)
    
    # Up-Gap Efficiency
    data['up_gap_eff'] = np.where(data['close'] > data['open'],
                                (data['close'] - data['open']) / data['gap_magnitude'], 0)
    
    # Down-Gap Efficiency
    data['down_gap_eff'] = np.where(data['close'] < data['open'],
                                  (data['open'] - data['close']) / data['gap_magnitude'], 0)
    
    # Asymmetric Gap Ratio
    data['asym_gap_ratio'] = data['up_gap_eff'] / (data['down_gap_eff'] + 1e-8)
    
    # Volume-Weighted Gap Momentum
    data['gap_momentum_intensity'] = (data['close'] - data['open']) / (data['gap_magnitude'] + 1e-8)
    data['volume_pressure'] = data['volume'] / (data['volume'].shift(1) + 1e-8) - 1
    data['volume_weighted_gap'] = data['gap_momentum_intensity'] * data['volume_pressure'] * np.sign(data['close'] - data['open'])
    
    # Fractal Gap Persistence Analysis
    data['gap_direction'] = np.sign(data['close'] - data['open'])
    
    # Gap Direction Persistence (rolling window of 5 days)
    gap_direction_persistence = []
    for i in range(len(data)):
        if i < 4:
            gap_direction_persistence.append(np.nan)
        else:
            window = data['gap_direction'].iloc[i-4:i+1]
            persistence = (window == window.shift(1)).sum() - 1  # Count consecutive same signs
            gap_direction_persistence.append(persistence)
    data['gap_dir_persistence'] = gap_direction_persistence
    
    # Gap Efficiency Persistence (rolling correlation)
    gap_eff_persistence = []
    for i in range(len(data)):
        if i < 5:
            gap_eff_persistence.append(np.nan)
        else:
            current_window = data['gap_momentum_intensity'].iloc[i-4:i+1]
            prev_window = data['gap_momentum_intensity'].iloc[i-5:i]
            if len(current_window) == len(prev_window) and len(current_window) > 1:
                corr = np.corrcoef(current_window, prev_window)[0, 1]
                gap_eff_persistence.append(corr if not np.isnan(corr) else 0)
            else:
                gap_eff_persistence.append(0)
    data['gap_eff_persistence'] = gap_eff_persistence
    
    # Fractal Gap Stability
    data['fractal_gap_stability'] = data['gap_dir_persistence'] * data['gap_eff_persistence']
    
    # Asymmetric Range Acceleration System
    # Multi-Scale Range Dynamics
    data['short_term_range_exp'] = (data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3) + 1e-8)
    data['medium_term_range_exp'] = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5) + 1e-8)
    data['range_accel_signal'] = (data['short_term_range_exp'] - 1) * (data['medium_term_range_exp'] - 1)
    
    # Fractal Range Utilization
    data['daily_range'] = data['high'] - data['low'] + 1e-8
    data['up_range_util'] = np.where(data['close'] > data['open'],
                                   (data['close'] - data['low']) / data['daily_range'], 0)
    data['down_range_util'] = np.where(data['close'] < data['open'],
                                     (data['high'] - data['close']) / data['daily_range'], 0)
    data['range_asym_ratio'] = data['up_range_util'] / (data['down_range_util'] + 1e-8)
    
    # Volume-Confirmed Acceleration
    data['volume_growth_accel'] = ((data['volume'] / (data['volume'].shift(1) + 1e-8) - 1) * 
                                 (data['volume'] / (data['volume'].shift(3) + 1e-8) - 1))
    data['range_volume_corr'] = data['range_accel_signal'] * data['volume_growth_accel']
    data['accel_confirmation'] = data['range_volume_corr'] * np.sign(data['close'] - data['open'])
    
    # Fractal Liquidity Pressure Anchoring
    # Liquidity Pressure Indicators
    data['price_impact_eff'] = np.abs(data['close'] - data['open']) / data['daily_range']
    data['volume_efficiency'] = data['volume'] / data['daily_range']
    data['liquidity_stress_div'] = data['price_impact_eff'] - data['volume_efficiency']
    
    # Fractal Reversion Signals
    data['short_term_reversion'] = ((data['close'] / (data['close'].shift(3) + 1e-8) - 1) * 
                                  (data['close'] / (data['close'].shift(5) + 1e-8) - 1))
    data['pressure_based_reversion'] = (-np.sign(data['close'] - data['open']) * 
                                      ((data['close'] - data['low']) / data['daily_range']))
    data['combined_reversion_force'] = data['short_term_reversion'] * data['pressure_based_reversion']
    
    # Volume-Weighted Anchoring
    data['volume_deviation'] = (data['volume'] - 
                              (data['volume'].shift(1) + data['volume'].shift(2) + 
                               data['volume'].shift(3) + data['volume'].shift(4) + 
                               data['volume'].shift(5)) / 5)
    data['anchoring_strength'] = data['volume_deviation'] * data['liquidity_stress_div']
    data['anchored_reversion'] = data['combined_reversion_force'] * data['anchoring_strength']
    
    # Fractal Regime Classification
    # Volatility Regime Classification
    data['fractal_vol_ratio'] = data['daily_range'] / (data['daily_range'].rolling(window=20).mean() + 1e-8)
    data['vol_regime_mult'] = np.where(data['fractal_vol_ratio'] > 1.5, 1.4,
                                     np.where(data['fractal_vol_ratio'] < 0.6, 0.6, 1.0))
    
    # Asymmetry Regime Classification
    data['asym_regime_mult'] = np.where(data['asym_gap_ratio'] > 1.5, 1.3,
                                      np.where(data['asym_gap_ratio'] < 0.67, 0.7, 1.0))
    
    # Fractal Regime Multiplier
    data['fractal_regime_mult'] = data['vol_regime_mult'] * data['asym_regime_mult']
    
    # Composite Alpha Generation
    data['core_gap_momentum'] = data['volume_weighted_gap'] * data['fractal_gap_stability']
    data['enhanced_range_accel'] = data['core_gap_momentum'] * data['range_accel_signal']
    data['liquidity_confirmed'] = data['enhanced_range_accel'] * data['accel_confirmation']
    data['regime_optimized'] = data['liquidity_confirmed'] * data['fractal_regime_mult']
    data['final_alpha'] = data['regime_optimized'] * data['anchored_reversion'] * data['combined_reversion_force']
    
    # Return the final alpha factor series
    return data['final_alpha']
