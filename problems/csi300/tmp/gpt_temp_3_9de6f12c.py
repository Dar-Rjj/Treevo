import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required shifts
    df['close_shift_1'] = df['close'].shift(1)
    df['close_shift_2'] = df['close'].shift(2)
    df['close_shift_5'] = df['close'].shift(5)
    df['close_shift_10'] = df['close'].shift(10)
    df['volume_shift_1'] = df['volume'].shift(1)
    df['volume_shift_5'] = df['volume'].shift(5)
    df['volume_shift_10'] = df['volume'].shift(10)
    df['high_shift_5'] = df['high'].shift(5)
    df['low_shift_5'] = df['low'].shift(5)
    
    for i in range(len(df)):
        if i < 10:  # Need at least 10 days of history
            result.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        
        # Quantum-Regime Detection Layer
        # Quantum Volatility Regime
        if current['high_shift_5'] != current['low_shift_5'] and current['volume_shift_5'] != 0:
            quantum_vol = ((current['high'] - current['low']) / (current['high_shift_5'] - current['low_shift_5']) * 
                          (current['volume'] / current['volume_shift_5']) * 
                          (current['high'] - current['open']) * (current['open'] - current['low']) / 
                          (current['close_shift_1'] ** 2 + 1e-8))
        else:
            quantum_vol = 0
            
        quantum_vol_flag = 1 if quantum_vol > np.percentile([quantum_vol], 50) else 0
        
        # Entropic Trend Regime
        if i >= 10:
            entropic_trend = (np.sign(current['close'] - current['close_shift_5']) * 
                             np.sign(current['close_shift_5'] - current['close_shift_10']) * 
                             abs((current['high'] - current['close']) * (current['close'] - current['low'])) / 
                             (current['close_shift_1'] ** 2 + 1e-8))
        else:
            entropic_trend = 0
            
        entropic_flag = 1 if entropic_trend > 0 else (-1 if entropic_trend < 0 else 0)
        
        # Quantum-Regime Adaptive Factors
        # High Quantum Volatility Momentum
        if current['high'] != current['low'] and current['volume_shift_1'] != 0:
            high_vol_momentum = ((current['close'] - current['open']) / (current['high'] - current['low']) * 
                                (current['volume'] / current['volume_shift_1']) * 
                                (current['high'] - current['open']) * (current['open'] - current['low']) / 
                                (current['close_shift_1'] ** 2 + 1e-8) * 
                                abs((current['close'] - current['close_shift_1']) / (current['high'] - current['low'] + 1e-8)))
        else:
            high_vol_momentum = 0
            
        # Low Quantum Volatility Mean Reversion
        if current['high'] != current['low'] and current['volume_shift_1'] != 0:
            low_vol_mean_rev = (((current['close'] - (current['high'] + current['low']) / 2) / (current['high'] - current['low'])) * 
                               (current['volume'] / current['volume_shift_1']) * 
                               np.sign(current['close'] - current['close_shift_1']) * 
                               (current['high'] - current['close']) * (current['close'] - current['low']) / 
                               (current['close_shift_1'] ** 2 + 1e-8))
        else:
            low_vol_mean_rev = 0
            
        # Bull Entropic Momentum Amplifier
        if current['high'] != current['low'] and current['volume'] != 0 and current['volume_shift_5'] != 0:
            bull_entropic_momentum = (((current['close'] - current['close_shift_1']) / (current['high'] - current['low'])) * 
                                     (current['amount'] / (current['volume'] + 1e-8)) * 
                                     (current['volume'] / current['volume_shift_5']) * 
                                     (current['high'] - current['open']) * (current['open'] - current['low']) / 
                                     (current['close_shift_1'] ** 2 + 1e-8))
        else:
            bull_entropic_momentum = 0
            
        # Bear Entropic Reversal Capture
        if current['high'] != current['low'] and current['amount'] != 0:
            bear_entropic_reversal = ((abs(current['open'] - current['close_shift_1']) / (current['high'] - current['low'])) * 
                                     (current['close'] - (current['high'] + current['low']) / 2) * 
                                     current['volume'] / (current['amount'] + 1e-8) * 
                                     (current['high'] - current['close']) * (current['close'] - current['low']) / 
                                     (current['close_shift_1'] ** 2 + 1e-8))
        else:
            bear_entropic_reversal = 0
            
        # Core Quantum Alpha
        if quantum_vol_flag == 1:
            core_quantum_alpha = high_vol_momentum
        else:
            core_quantum_alpha = low_vol_mean_rev
            
        if entropic_flag == 1:
            core_quantum_alpha += bull_entropic_momentum
        elif entropic_flag == -1:
            core_quantum_alpha += bear_entropic_reversal
            
        # Quantum-Regime Interaction
        # Volatility-Entropy Convergence
        vol_entropy_conv = (high_vol_momentum * bull_entropic_momentum - 
                           low_vol_mean_rev * bear_entropic_reversal) * \
                          (current['volume'] / (current['volume_shift_1'] + 1e-8)) * \
                          np.sign((current['high'] - current['open']) * (current['open'] - current['low']))
        
        # Multi-Timeframe Quantum Integration
        # Short-Term Quantum
        if current['high'] != current['low'] and current['volume_shift_1'] != 0:
            short_term_quantum = ((current['close'] - current['open']) / (current['high'] - current['low']) * 
                                 current['volume'] / current['volume_shift_1'] * 
                                 (current['high'] - current['open']) * (current['open'] - current['low']) / 
                                 (current['close_shift_1'] ** 2 + 1e-8))
        else:
            short_term_quantum = 0
            
        # Medium-Term Quantum
        if current['high_shift_5'] != current['low_shift_5'] and current['volume_shift_5'] != 0:
            medium_term_quantum = (((current['close'] - current['close_shift_5']) / (current['high_shift_5'] - current['low_shift_5'])) * 
                                  current['volume'] / current['volume_shift_5'] * 
                                  abs((current['high'] - current['close']) * (current['close'] - current['low'])) / 
                                  (current['close_shift_1'] ** 2 + 1e-8))
        else:
            medium_term_quantum = 0
            
        # Long-Term Quantum
        if current['volume_shift_10'] != 0 and abs(current['close_shift_5'] - current['close_shift_10']) > 1e-8:
            long_term_quantum = (np.sign(current['close'] - current['close_shift_10']) * 
                                abs(current['close'] - current['close_shift_5']) / abs(current['close_shift_5'] - current['close_shift_10']) * 
                                current['volume'] / current['volume_shift_10'] * 
                                (current['high'] - current['close']) * (current['close'] - current['low']) / 
                                (current['close_shift_1'] ** 2 + 1e-8))
        else:
            long_term_quantum = 0
            
        multi_timeframe = (short_term_quantum + medium_term_quantum + long_term_quantum) / 3
        
        # Timing Quantum Alpha
        timing_quantum_alpha = vol_entropy_conv * multi_timeframe
        
        # Quantum Validation Enhancement
        entropic_efficiency = np.sign((current['high'] - current['open']) * (current['open'] - current['low']) / 
                                     (current['close_shift_1'] ** 2 + 1e-8) - 1) * \
                            np.sign((current['close'] - current['close_shift_1']) / (current['high'] - current['low'] + 1e-8))
        
        volume_quantum_align = np.sign(current['volume'] / (current['volume_shift_1'] + 1e-8) - 1) * \
                              np.sign((current['high'] - current['open']) * (current['open'] - current['low']))
        
        quantum_temporal_triple = entropic_efficiency * volume_quantum_align
        
        # Validation boosts
        validation_boost = 1.0
        if entropic_efficiency > 0:
            validation_boost *= 1.25
        if volume_quantum_align > 0:
            validation_boost *= 1.15
        if quantum_temporal_triple > 0:
            validation_boost *= 1.3
            
        # Asymmetric Quantum Response
        if current['close'] > current['close_shift_2']:
            up_move_factor = high_vol_momentum
            down_move_factor = 0
        else:
            up_move_factor = 0
            down_move_factor = low_vol_mean_rev
            
        quantum_asymmetry_ratio = up_move_factor / (down_move_factor + 1e-8)
        
        # Asymmetric Adjustment
        asymmetric_adjustment = up_move_factor if current['close'] > current['close_shift_2'] else down_move_factor
        
        # Final Quantum-Regime Alpha
        final_alpha = ((core_quantum_alpha * timing_quantum_alpha + asymmetric_adjustment) * 
                      quantum_asymmetry_ratio * validation_boost)
        
        result.iloc[i] = final_alpha
        
    return result
