import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Add small epsilon to avoid division by zero
    eps = 0.001
    
    # Multi-Scale Range Momentum Structure
    # Micro Range Momentum
    micro_range_momentum = ((data['close'] - data['close'].shift(1)) / 
                           (data['high'] - data['low'] + eps) * 
                           (data['high'] - data['low']) / 
                           (data['high'].shift(1) - data['low'].shift(1) + eps))
    
    # Meso Range Momentum
    high_3d = data['high'].rolling(window=3, min_periods=3).max()
    low_3d = data['low'].rolling(window=3, min_periods=3).min()
    high_prev_3d = data['high'].shift(3).rolling(window=3, min_periods=3).max()
    low_prev_3d = data['low'].shift(3).rolling(window=3, min_periods=3).min()
    
    meso_range_momentum = ((data['close'] - data['close'].shift(3)) / 
                          (high_3d - low_3d + eps) * 
                          (high_3d - low_3d) / 
                          (high_prev_3d - low_prev_3d + eps))
    
    # Macro Range Momentum
    high_8d = data['high'].rolling(window=8, min_periods=8).max()
    low_8d = data['low'].rolling(window=8, min_periods=8).min()
    high_prev_8d = data['high'].shift(8).rolling(window=8, min_periods=8).max()
    low_prev_8d = data['low'].shift(8).rolling(window=8, min_periods=8).min()
    
    macro_range_momentum = ((data['close'] - data['close'].shift(8)) / 
                           (high_8d - low_8d + eps) * 
                           (high_8d - low_8d) / 
                           (high_prev_8d - low_prev_8d + eps))
    
    # Volume-Range Asymmetric Dynamics
    # Range Buy/Sell Pressure
    range_buy_pressure = (data['volume'] * 
                         (data['close'] - data['low']) / (data['high'] - data['low'] + eps) * 
                         (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + eps))
    
    range_sell_pressure = (data['volume'] * 
                          (data['high'] - data['close']) / (data['high'] - data['low'] + eps) * 
                          (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + eps))
    
    range_pressure_asymmetry = ((range_buy_pressure - range_sell_pressure) * 
                               np.sign(range_buy_pressure - range_sell_pressure).shift(1))
    
    # Volume-Range Flow Quality
    volume_range_trend_short = ((data['volume'] / data['volume'].shift(1)) * 
                               ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + eps)))
    
    volume_range_trend_medium = ((data['volume'] / data['volume'].shift(3)) * 
                                ((high_3d - low_3d) / (high_prev_3d - low_prev_3d + eps)))
    
    volume_range_trend_long = ((data['volume'] / data['volume'].shift(8)) * 
                              ((high_8d - low_8d) / (high_prev_8d - low_prev_8d + eps)))
    
    volume_range_flow_direction = np.sign(volume_range_trend_short + volume_range_trend_medium + volume_range_trend_long)
    
    # Volume-Range Concentration Patterns
    range_morning_concentration = (data['volume'] * 
                                  (data['close'] - data['open']) / (data['high'] - data['low'] + eps) * 
                                  (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + eps))
    
    range_afternoon_concentration = (data['volume'] * 
                                    (data['open'] - data['close']) / (data['high'] - data['low'] + eps) * 
                                    (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + eps))
    
    range_concentration_differential = range_morning_concentration - range_afternoon_concentration
    
    # Fractal Volatility-Momentum Integration
    # Multi-Scale Volatility Patterns
    micro_range_volatility = ((data['high'] - data['low']) / data['close'] * 
                             (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + eps))
    
    meso_range_volatility = ((high_3d - low_3d) / data['close'].shift(3) * 
                            (high_3d - low_3d) / (high_prev_3d - low_prev_3d + eps))
    
    macro_range_volatility = ((high_8d - low_8d) / data['close'].shift(8) * 
                             (high_8d - low_8d) / (high_prev_8d - low_prev_8d + eps))
    
    # Volatility-Momentum Synchronization
    volatility_adjusted_flow_asymmetry = ((data['high'] - data['open']) / (data['high'] - data['low'] + eps) - 
                                         (data['close'] - data['low']) / (data['high'] - data['low'] + eps) * 
                                         (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5) + eps))
    
    fractal_price_impact_momentum = ((data['close'] - data['open']) / data['volume'] * 
                                    (data['close'] - data['close'].shift(1)) / 
                                    (np.abs(data['close'] - data['close'].shift(1)) + eps))
    
    # Volume-Flow Regime Persistence
    close_open_diff = data['close'] - data['open']
    up_count = close_open_diff.rolling(window=5, min_periods=5).apply(lambda x: (x > 0).sum())
    down_count = close_open_diff.rolling(window=5, min_periods=5).apply(lambda x: (x < 0).sum())
    volume_flow_regime_persistence = ((up_count - down_count) * 
                                     (data['volume'] - data['volume'].shift(5)) / 
                                     (data['volume'].shift(5) + eps))
    
    # Range Volatility Regime Detection
    micro_gt_meso = (micro_range_volatility > meso_range_volatility).rolling(window=3, min_periods=3).sum()
    micro_lt_meso = (micro_range_volatility < meso_range_volatility).rolling(window=3, min_periods=3).sum()
    range_volatility_regime_strength = micro_gt_meso - micro_lt_meso
    
    # Path Efficiency & Reversal Analysis
    # Range-Path Efficiency Integration
    micro_range_path_efficiency = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + eps) * 
                                  (data['close'] - data['close'].shift(2)) / 
                                  (np.abs(data['close'] - data['close'].shift(1)) + 
                                   np.abs(data['close'].shift(1) - data['close'].shift(2)) + eps))
    
    high_5d = data['high'].rolling(window=5, min_periods=5).max()
    low_5d = data['low'].rolling(window=5, min_periods=5).min()
    
    meso_range_path_integration = ((data['close'] - data['close'].shift(5)) / (high_5d - low_5d + eps) * 
                                  (data['close'] - data['close'].shift(3)) / 
                                  (np.abs(data['close'] - data['close'].shift(1)) + 
                                   np.abs(data['close'].shift(1) - data['close'].shift(2)) + 
                                   np.abs(data['close'].shift(2) - data['close'].shift(3)) + eps))
    
    range_path_efficiency_score = micro_range_path_efficiency * meso_range_path_integration
    
    # Fractal Reversal Patterns
    range_price_reversal = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + eps) * 
                           (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + eps))
    
    high_2d = data['high'].rolling(window=2, min_periods=2).max()
    low_2d = data['low'].rolling(window=2, min_periods=2).min()
    high_prev_2d = data['high'].shift(2).rolling(window=2, min_periods=2).max()
    low_prev_2d = data['low'].shift(2).rolling(window=2, min_periods=2).min()
    
    range_price_fractal_momentum = ((data['close'] - data['open']) / (high_2d - low_2d + eps) * 
                                   (high_2d - low_2d) / (high_prev_2d - low_prev_2d + eps))
    
    # Reversal Persistence
    close_diff_pos = (data['close'] - data['close'].shift(1) > 0).rolling(window=3, min_periods=3).sum()
    avg_range_3d = (data['high'] - data['low']).rolling(window=3, min_periods=3).mean()
    reversal_persistence = close_diff_pos / ((data['high'] - data['low']) / (avg_range_3d + eps) + eps)
    
    # Fractal Momentum Enhancement
    # Volume-Weighted Momentum Transmission
    volume_asymmetry = (data['volume'] * 
                       (data['close'] - data['open']) / (data['high'] - data['low'] + eps) * 
                       (data['high'] - data['open'] - (data['open'] - data['low'])))
    
    volume_timing = (data['volume'] * (data['open'] - data['close'].shift(1)) / 
                    (np.abs(data['open'] - data['close'].shift(1)) + eps) - 
                    data['volume'] * (data['close'] - data['open']) / 
                    (np.abs(data['close'] - data['open']) + eps))
    
    volume_range_coupling = ((data['volume'] / (data['high'] - data['low'] + eps)) * 
                            (data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2)) / 2 + eps)) * 
                            (data['close'] - data['open']) / (data['high'] - data['low'] + eps))
    
    # Fractal Momentum Pressure
    morning_momentum = (data['high'] - np.maximum(data['open'], data['close'])) * (data['close'] - data['low'])
    afternoon_momentum = (np.minimum(data['open'], data['close']) - data['low']) * (data['high'] - data['close'])
    price_fractal_momentum = (data['close'] - data['open']) / (high_2d - low_2d + eps)
    
    # Enhanced Range Momentum
    range_accelerated_momentum = (data['close'] - data['close'].shift(1)) * volume_range_trend_short
    range_sustained_momentum = (data['close'] - data['close'].shift(3)) * volume_range_trend_medium
    range_momentum_quality = np.sign(range_accelerated_momentum) * np.sign(range_sustained_momentum)
    
    # Regime-Adaptive Synchronization
    # Range Volatility Regime Components
    fractal_range_momentum_strength = micro_range_momentum + meso_range_momentum + macro_range_momentum
    range_efficiency_momentum = range_path_efficiency_score * volatility_adjusted_flow_asymmetry
    
    range_high_volatility_alpha = fractal_range_momentum_strength * range_volatility_regime_strength
    range_low_volatility_alpha = range_efficiency_momentum * (1 - np.abs(range_volatility_regime_strength))
    range_volatility_adaptive = range_high_volatility_alpha + range_low_volatility_alpha
    
    # Volume Regime Components
    volume_trend_consistency = ((np.sign(volume_range_trend_short) == np.sign(volume_range_trend_medium)) & 
                               (np.sign(volume_range_trend_medium) == np.sign(volume_range_trend_long)))
    volume_trend_consistency_count = volume_trend_consistency.rolling(window=3, min_periods=3).sum()
    
    range_volume_momentum = volume_asymmetry * volume_timing * volume_range_coupling
    range_volume_driven_alpha = range_volume_momentum * volume_trend_consistency_count
    range_volume_adaptive = range_volume_driven_alpha * volume_range_flow_direction
    
    # Multi-Dimensional Convergence
    signal_convergence = (np.sign((data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'] + eps) - 
                                 (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'] + eps)) * 
                         np.sign(data['amount'] * (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'] + eps) - 
                                 data['amount'] * (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'] + eps)))
    
    range_efficiency_momentum_convergence = np.sign(range_efficiency_momentum) * np.sign(fractal_range_momentum_strength)
    multi_dimensional_alignment = signal_convergence * range_efficiency_momentum_convergence
    
    # Trade Size Momentum Enhancement
    trade_size_momentum = (data['amount'] / data['volume']) / (data['amount'].shift(1) / data['volume'].shift(1) + eps)
    
    # Final Alpha Construction
    # Core Synchronization Components
    range_momentum_core = ((micro_range_momentum + meso_range_momentum + macro_range_momentum) * 
                          range_momentum_quality * range_pressure_asymmetry)
    
    volume_transmission_core = (volume_asymmetry * volume_timing * volume_range_coupling * trade_size_momentum)
    
    efficiency_core = (range_path_efficiency_score * volatility_adjusted_flow_asymmetry * 
                      volume_flow_regime_persistence)
    
    # Regime Integration
    volatility_regime_weight = range_volatility_adaptive * range_volatility_regime_strength
    volume_regime_weight = range_volume_adaptive * volume_trend_consistency_count
    reversal_enhancement = range_price_reversal * reversal_persistence
    
    # Final Alpha Output
    alpha = (range_momentum_core * volume_transmission_core * efficiency_core * 
             volatility_regime_weight * volume_regime_weight * reversal_enhancement * 
             multi_dimensional_alignment)
    
    return alpha
