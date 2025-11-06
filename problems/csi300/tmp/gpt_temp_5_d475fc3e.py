import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['gap_open_close'] = abs(df['open'] - df['close'].shift(1))
    df['daily_range'] = df['high'] - df['low']
    df['gap_ratio'] = df['gap_open_close'] / df['daily_range']
    df['gap_direction'] = np.sign(df['open'] - df['close'].shift(1))
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['close_change'] = df['close'] - df['close'].shift(1)
    df['amount_direction'] = np.sign(df['close'] - df['open'])
    
    # Rolling calculations
    for i in range(len(df)):
        if i < 14:  # Need enough history
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        
        # Quantum Gap Efficiency
        close_5_diff = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-5])
        high_4_max = current_data['high'].iloc[i-4:i+1].max()
        low_4_min = current_data['low'].iloc[i-4:i+1].min()
        range_4 = high_4_max - low_4_min if high_4_max != low_4_min else 1
        quantum_gap_efficiency = (close_5_diff / range_4) * current_data['gap_ratio'].iloc[i]
        
        # Entropic Gap Momentum
        close_10_diff = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-10])
        high_9_max = current_data['high'].iloc[i-9:i+1].max()
        low_9_min = current_data['low'].iloc[i-9:i+1].min()
        range_9 = high_9_max - low_9_min if high_9_max != low_9_min else 1
        efficiency_10 = (close_10_diff / range_9) * current_data['gap_ratio'].iloc[i]
        
        efficiency_5 = quantum_gap_efficiency
        momentum_diff = efficiency_10 - efficiency_5
        entropic_gap_momentum = (momentum_diff * np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-2]) * 
                                current_data['gap_ratio'].iloc[i] / current_data['close'].shift(1).iloc[i])
        
        # Bidirectional Gap Entropy
        bidirectional_gap_entropy = (current_data['gap_ratio'].iloc[i] * current_data['gap_direction'].iloc[i] * 
                                   current_data['volume_ratio'].iloc[i])
        
        # Quantum Gap Compression
        high_3_max = current_data['high'].iloc[i-3:i+1].max()
        low_3_min = current_data['low'].iloc[i-3:i+1].min()
        high_6_max = current_data['high'].iloc[i-6:i+1].max()
        low_6_min = current_data['low'].iloc[i-6:i+1].min()
        range_3 = high_3_max - low_3_min if high_3_max != low_3_min else 1
        range_6 = high_6_max - low_6_min if high_6_max != low_6_min else 1
        quantum_gap_compression = (range_3 / range_6) * current_data['gap_ratio'].iloc[i]
        
        # Quantum Gap Flow Imbalance
        amount_7 = current_data['amount'].iloc[i-6:i+1]
        positive_amount = amount_7[amount_7 > 0].sum()
        negative_amount = abs(amount_7[amount_7 < 0]).sum()
        total_amount = amount_7.abs().sum()
        flow_imbalance = ((positive_amount - negative_amount) / total_amount if total_amount > 0 else 0) * current_data['gap_ratio'].iloc[i]
        
        # Entropic Gap Volume Pressure
        entropic_gap_volume_pressure = (current_data['volume_ratio'].iloc[i] * current_data['amount_direction'].iloc[i] * 
                                      current_data['gap_ratio'].iloc[i])
        
        # Quantum Gap Flow Integration
        quantum_gap_flow_integration = flow_imbalance * entropic_gap_volume_pressure * np.sign(current_data['close_change'].iloc[i])
        
        # Entropic Gap Fracture
        open_low_diff = current_data['open'].iloc[i] - current_data['low'].iloc[i]
        high_open_diff = current_data['high'].iloc[i] - current_data['open'].iloc[i]
        close_change_sign = np.sign(current_data['close_change'].iloc[i]) if current_data['close_change'].iloc[i] != 0 else 1
        entropic_gap_fracture = ((open_low_diff - high_open_diff) * close_change_sign * current_data['gap_ratio'].iloc[i])
        
        # Quantum Gap Closing Momentum
        high_2_max = current_data['high'].iloc[i-2:i+1].max()
        low_2_min = current_data['low'].iloc[i-2:i+1].min()
        range_2 = high_2_max - low_2_min if high_2_max != low_2_min else 1
        quantum_gap_closing_momentum = ((current_data['close'].iloc[i] - current_data['open'].iloc[i]) / range_2 * 
                                      current_data['volume_ratio'].iloc[i])
        
        # Quantum Gap Volume Asymmetry
        volume_7 = current_data['volume'].iloc[i-6:i+1]
        close_7 = current_data['close'].iloc[i-6:i+1]
        open_7 = current_data['open'].iloc[i-6:i+1]
        up_volume = volume_7[close_7 > open_7].sum()
        down_volume = volume_7[close_7 < open_7].sum()
        volume_asymmetry = (up_volume / down_volume if down_volume > 0 else 1) * np.sign(current_data['volume'].iloc[i] - current_data['volume'].iloc[i-3]) * current_data['volume_ratio'].iloc[i]
        
        # Regime Classification
        high_entropic_divergence = (abs(bidirectional_gap_entropy) > 0.7 and entropic_gap_momentum > 0 and quantum_gap_efficiency > 1.1)
        low_entropic_convergence = (abs(bidirectional_gap_entropy) < 0.5 and entropic_gap_momentum < 0 and quantum_gap_efficiency < 0.9)
        momentum_clustering = (quantum_gap_closing_momentum > 1.1 and volume_asymmetry > 1.0)
        flow_divergence = (flow_imbalance > 1.2 and volume_asymmetry < 0.8)
        
        # Regime-Adaptive Synthesis
        if high_entropic_divergence:
            core_signal = entropic_gap_momentum * quantum_gap_flow_integration * quantum_gap_efficiency
            enhancement = core_signal * flow_imbalance * (1 - quantum_gap_compression)
            signal = enhancement
        elif low_entropic_convergence:
            core_signal = entropic_gap_fracture * quantum_gap_closing_momentum
            risk_adjustment = core_signal * (1 - abs(flow_imbalance))
            signal = risk_adjustment
        elif momentum_clustering:
            core_signal = quantum_gap_closing_momentum * volume_asymmetry * flow_imbalance
            momentum_enhancement = core_signal * (1 + quantum_gap_efficiency)
            signal = momentum_enhancement
        elif flow_divergence:
            core_signal = -1 * flow_imbalance / (1 + quantum_gap_efficiency)
            flow_adjustment = core_signal * volume_asymmetry
            signal = flow_adjustment
        else:  # Balanced regime
            price_component = 0.3 * entropic_gap_momentum * quantum_gap_closing_momentum
            volume_component = 0.3 * volume_asymmetry * flow_imbalance
            efficiency_component = 0.4 * quantum_gap_efficiency * (1 - abs(flow_imbalance))
            signal = price_component + volume_component + efficiency_component
        
        # Apply multipliers and adjustments
        if quantum_gap_efficiency > 1:
            multiplier = 1 + (quantum_gap_efficiency - 1) * 0.15
        elif quantum_gap_efficiency < 1:
            multiplier = 1 + (1 - quantum_gap_efficiency) * 0.08
        else:
            multiplier = 1 + abs(quantum_gap_efficiency - 1) * 0.03
        
        final_signal = signal * multiplier
        
        result.iloc[i] = final_signal
    
    return result
