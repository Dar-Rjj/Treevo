import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Helper functions
    def safe_divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    
    # Cross-Fracture Momentum components
    # Micro Cross-Fracture
    micro_momentum = safe_divide(data['close'] - data['close'].shift(1), 
                                data['high'] - data['low']) * \
                    safe_divide(data['volume'], data['volume'].shift(1)) * \
                    safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                               data['high'] - data['low'])
    
    # Meso Cross-Fracture
    high_5d = data['high'].rolling(window=6, min_periods=1).max()
    low_5d = data['low'].rolling(window=6, min_periods=1).min()
    meso_momentum = safe_divide(data['close'] - data['close'].shift(5), 
                               high_5d - low_5d) * \
                   safe_divide(data['volume'], data['volume'].shift(5)) * \
                   safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                              data['high'] - data['low'])
    
    # Macro Cross-Fracture
    high_21d = data['high'].rolling(window=22, min_periods=1).max()
    low_21d = data['low'].rolling(window=22, min_periods=1).min()
    macro_momentum = safe_divide(data['close'] - data['close'].shift(21), 
                                high_21d - low_21d) * \
                    safe_divide(data['volume'], data['volume'].shift(21)) * \
                    safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                               data['high'] - data['low'])
    
    # Cross-Fracture Divergence components
    # Opening Cross-Fracture
    opening_divergence = np.sign(data['open'] - data['close'].shift(1)) * \
                        safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                                   data['high'] - data['low']) * \
                        safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                                   np.abs(data['close'].shift(1) - data['close'].shift(2)))
    
    # Upper and Lower Cross-Fracture Efficiency for Intraday
    upper_efficiency = safe_divide(data['high'] - data['close'], 
                                  data['high'] - data['open'] + 1e-6)
    lower_efficiency = safe_divide(data['close'] - data['low'], 
                                  data['open'] - data['low'] + 1e-6)
    intraday_divergence = (upper_efficiency - lower_efficiency) * \
                         safe_divide(np.abs(data['close'] - data['open']), 
                                    data['high'] - data['low'])
    
    # Closing Cross-Fracture
    closing_divergence = (data['close'] - (data['high'] + data['low']) / 2) * \
                        safe_divide(np.abs(data['close'] - data['open']), 
                                   data['high'] - data['low']) * \
                        safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                                   data['high'] - data['low'])
    
    # Cross-Fracture Alignment
    cross_fracture_alignment = (micro_momentum + meso_momentum + macro_momentum) / 3 - \
                              (opening_divergence + intraday_divergence + closing_divergence) / 3
    
    # Quantum Efficiency Patterns
    # Gap Efficiency Cross-Fracture
    opening_efficiency = safe_divide(data['close'] - data['open'], 
                                    data['high'] - data['low']) * \
                       safe_divide(data['open'] - data['close'].shift(1), 
                                  np.abs(data['open'] - data['close'].shift(2))) * \
                       safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                                  data['high'] - data['low'])
    
    upper_efficiency_cf = upper_efficiency * \
                         safe_divide(np.abs(data['close'] - data['open']), 
                                    data['high'] - data['low']) * \
                         safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                                    data['high'] - data['low'])
    
    lower_efficiency_cf = lower_efficiency * \
                         safe_divide(np.abs(data['close'] - data['open']), 
                                    data['high'] - data['low']) * \
                         safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                                    data['high'] - data['low'])
    
    # Volume Efficiency Cross-Fracture
    opening_volume_efficiency = safe_divide(data['volume'], 
                                           np.maximum(data['volume'].shift(1), data['volume'].shift(2))) * \
                              safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                                         np.abs(data['close'].shift(1) - data['close'].shift(2))) * \
                              safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                                         data['high'] - data['low'])
    
    volume_5d_avg = data['volume'].rolling(window=5, min_periods=1).mean()
    closing_volume_efficiency = safe_divide(data['volume'], volume_5d_avg + 1e-6) * \
                              safe_divide(np.abs(data['close'] - data['open']), 
                                         data['high'] - data['low']) * \
                              safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                                         data['high'] - data['low'])
    
    amount_efficiency = safe_divide(data['amount'], 
                                   data['volume'] * data['close']) * \
                      safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                                 data['high'] - data['low'])
    
    # Efficiency Divergence
    efficiency_divergence = (opening_efficiency - (upper_efficiency_cf + lower_efficiency_cf) / 2) * \
                          safe_divide(data['volume'], data['volume'].shift(3))
    
    # Cross-Fracture Regime Adaptation
    # Volatility Regimes
    daily_vol = safe_divide(data['high'] - data['low'], data['close'].shift(1))
    gap_ratio = safe_divide(np.abs(data['open'] - data['close'].shift(1)), data['high'] - data['low'])
    
    high_vol_regime = (daily_vol > 0.04) & (gap_ratio > 0.3)
    low_vol_regime = (daily_vol < 0.01) & (gap_ratio < 0.1)
    moderate_vol_regime = ~high_vol_regime & ~low_vol_regime
    
    # Volume Regimes
    close_open_ratio = safe_divide(np.abs(data['close'] - data['open']), data['high'] - data['low'])
    high_volume_regime = (data['volume'] > 1.8 * data['volume'].shift(1)) & (close_open_ratio > 0.6)
    low_volume_regime = (data['volume'] < 0.6 * data['volume'].shift(1)) & (close_open_ratio < 0.4)
    normal_volume_regime = ~high_volume_regime & ~low_volume_regime
    
    # Efficiency Regimes
    high_efficiency_regime = (upper_efficiency > 0.7) & (lower_efficiency > 0.7)
    low_efficiency_regime = (upper_efficiency < 0.3) | (lower_efficiency < 0.3)
    moderate_efficiency_regime = ~high_efficiency_regime & ~low_efficiency_regime
    
    # Quantum Momentum Construction
    # Volatility-Weighted Momentum
    volatility_weighted_momentum = pd.Series(index=data.index, dtype=float)
    volatility_weighted_momentum[high_vol_regime] = micro_momentum[high_vol_regime] * 0.7 + meso_momentum[high_vol_regime] * 0.3
    volatility_weighted_momentum[low_vol_regime] = meso_momentum[low_vol_regime] * 0.6 + macro_momentum[low_vol_regime] * 0.4
    volatility_weighted_momentum[moderate_vol_regime] = (micro_momentum[moderate_vol_regime] + meso_momentum[moderate_vol_regime] + macro_momentum[moderate_vol_regime]) / 3
    
    # Volume-Enhanced Momentum
    volume_enhanced_momentum = volatility_weighted_momentum * (1 + opening_volume_efficiency * closing_volume_efficiency)
    
    # Efficiency-Enhanced Momentum
    gap_enhanced_momentum = volume_enhanced_momentum * (1 + opening_efficiency * safe_divide(np.abs(data['open'] - data['close'].shift(1)), data['high'] - data['low']))
    range_enhanced_momentum = gap_enhanced_momentum * (1 + (upper_efficiency_cf + lower_efficiency_cf) / 2 * safe_divide(np.abs(data['close'] - data['open']), data['high'] - data['low']))
    amount_enhanced_momentum = range_enhanced_momentum * (1 + amount_efficiency * safe_divide(data['volume'], data['volume'].shift(1)))
    
    # Cross-Fracture Alignment Momentum
    def directional_persistence(series, window=5):
        result = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i+1]
            current_sign = np.sign(series.iloc[i])
            persistence = (np.sign(window_data) == current_sign).sum()
            result.iloc[i] = persistence
        return result
    
    directional_persistence_val = directional_persistence(cross_fracture_alignment)
    volume_aligned_momentum = amount_enhanced_momentum * (1 + cross_fracture_alignment * directional_persistence_val)
    efficiency_aligned_momentum = volume_aligned_momentum * (1 + efficiency_divergence * safe_divide(np.abs(cross_fracture_alignment), data['high'] - data['low']))
    
    # Quantum Convergence Factors
    # Multi-Frequency Cross-Fracture
    high_frequency = safe_divide(np.abs((data['close'] - data['close'].shift(1)) - (data['close'].shift(1) - data['close'].shift(2))), 
                                data['high'] - data['low']) * \
                    safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                               data['high'] - data['low'])
    
    medium_frequency = safe_divide(np.abs(data['close'] - data['close'].shift(5)), 
                                  data['close'].diff().abs().rolling(window=5, min_periods=1).sum()) * \
                     safe_divide(data['high'] - data['low'], 
                                np.abs(data['open'] - data['close'].shift(1)))
    
    cross_frequency_momentum = high_frequency * medium_frequency * \
                              safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                                         data['high'] - data['low'])
    
    # Volume-Price Cross-Fracture
    bullish_divergence = ((data['close'] > data['open']) & (data['volume'] < data['volume'].shift(1))) * \
                        safe_divide(np.abs(data['close'] - data['open']), 
                                   data['high'] - data['low'])
    
    bearish_divergence = ((data['close'] < data['open']) & (data['volume'] > data['volume'].shift(1))) * \
                        safe_divide(np.abs(data['open'] - data['close']), 
                                   data['high'] - data['low'])
    
    volume_price_alignment = (bullish_divergence - bearish_divergence) * \
                           safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                                      data['high'] - data['low'])
    
    # Volatility Cross-Fracture
    volatility_skew = safe_divide(data['high'] - data['open'] - (data['open'] - data['low']), 
                                 np.abs(data['open'] - data['close'].shift(1))) * \
                     safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                                data['high'] - data['low'])
    
    regime_shift = safe_divide(data['high'] - data['low'], data['high'].shift(3) - data['low'].shift(3)) - \
                  safe_divide(data['high'] - data['low'], data['high'].shift(15) - data['low'].shift(15))
    
    volatility_momentum = volatility_skew * regime_shift * \
                        safe_divide(np.abs(data['open'] - data['close'].shift(1)), 
                                   data['high'] - data['low'])
    
    # Quantum Convergence
    quantum_convergence = cross_frequency_momentum * volume_price_alignment * volatility_momentum
    
    # Regime Multipliers
    volatility_multiplier = pd.Series(1.0, index=data.index)
    volatility_multiplier[high_vol_regime] = 0.8
    volatility_multiplier[low_vol_regime] = 1.2
    
    volume_multiplier = pd.Series(1.0, index=data.index)
    volume_multiplier[high_volume_regime] = 1.1
    volume_multiplier[low_volume_regime] = 0.9
    
    efficiency_multiplier = pd.Series(1.0, index=data.index)
    efficiency_multiplier[high_efficiency_regime] = 1.15
    efficiency_multiplier[low_efficiency_regime] = 0.85
    
    # Final Quantum Cross-Fracture Alpha
    base_alpha = efficiency_aligned_momentum
    final_alpha = base_alpha * (1 + quantum_convergence) * volatility_multiplier * volume_multiplier * efficiency_multiplier
    
    return final_alpha
