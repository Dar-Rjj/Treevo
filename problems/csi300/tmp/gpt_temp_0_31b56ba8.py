import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 5:  # Need at least 5 days of data for calculations
            result.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        prev1 = df.iloc[i-1]
        prev2 = df.iloc[i-2]
        prev3 = df.iloc[i-3]
        prev4 = df.iloc[i-4]
        
        # Quantum State Preparation components
        # Price State Preparation
        quantum_price_basis = (current['high'] + current['low']) / (2 * current['close']) - 1
        price_state_purity = abs(current['close'] - (current['open'] + current['high'] + current['low'])/3) / (current['high'] - current['low'])
        
        state_prep_fidelity = 0
        for j in range(5):
            if i-j-1 >= 0:
                state_prep_fidelity += (df.iloc[i-j]['close'] - df.iloc[i-j-1]['close'])**2
        state_prep_fidelity /= (current['high'] - current['low'])**2
        
        # Volume State Preparation
        volume_diff_sum = 0
        for j in range(5):
            if i-j-1 >= 0:
                volume_diff_sum += abs(df.iloc[i-j]['volume'] - df.iloc[i-j-1]['volume'])
        volume_state_init = current['volume'] / volume_diff_sum if volume_diff_sum != 0 else 0
        
        volume_state_transform = (current['volume'] - prev1['volume']) / abs(prev1['volume'] - prev2['volume']) if abs(prev1['volume'] - prev2['volume']) != 0 else 0
        prep_quality = abs(current['volume'] - prev1['volume']) / prev1['volume'] if prev1['volume'] != 0 else 0
        
        # Joint State Preparation
        prep_strength = (current['close'] - current['open']) * np.log(current['volume'] + 1)
        prep_correlation = np.sign(current['close'] - prev1['close']) * np.sign(current['volume'] - prev1['volume'])
        
        prep_consistency = 0
        for j in range(5):
            if i-j-1 >= 0:
                prep_consistency += np.sign(df.iloc[i-j]['close'] - df.iloc[i-j-1]['close']) * np.sign(df.iloc[i-j]['volume'] - df.iloc[i-j-1]['volume'])
        
        # Quantum Channel Operations
        # Price Channel Operations
        quantum_price_channel = (current['high'] - current['low']) / abs(current['close'] - prev1['close']) if abs(current['close'] - prev1['close']) != 0 else 0
        channel_energy = (current['close'] - current['open'])**2 / (current['high'] - current['low']) if (current['high'] - current['low']) != 0 else 0
        channel_stability = abs(current['close'] - prev1['close']) / (prev1['high'] - prev1['low']) if (prev1['high'] - prev1['low']) != 0 else 0
        
        # Volume Channel Operations
        volume_sum = sum(df.iloc[i-j]['volume'] for j in range(1, 5))
        volume_channel_capacity = current['volume'] / volume_sum if volume_sum != 0 else 0
        quantum_volume_processing = (current['volume'] - prev1['volume']) / prev2['volume'] if prev2['volume'] != 0 else 0
        
        volume_diff_sum2 = sum(abs(df.iloc[i-j]['volume'] - df.iloc[i-j-1]['volume']) for j in range(5))
        channel_coherence = volume_diff_sum2 / current['volume'] if current['volume'] != 0 else 0
        
        # Multi-Channel Interactions
        channel_coupling = (current['high'] - current['low']) * np.sqrt(current['volume'])
        quantum_channel_interference = (current['close'] - prev1['close']) * (current['volume'] - prev1['volume'])
        channel_resonance = (current['close'] - current['open']) / (current['volume']**(1/3)) if current['volume'] != 0 else 0
        
        # Quantum Error Correction
        # Price Error Detection
        quantum_price_error = abs((current['close'] - prev1['close']) - (prev1['close'] - prev2['close'])) / (current['high'] - current['low']) if (current['high'] - current['low']) != 0 else 0
        error_detection_threshold = (current['high'] - current['low']) / abs(current['close'] - prev1['close']) if abs(current['close'] - prev1['close']) != 0 else 0
        
        close_diff_sum = sum(abs(df.iloc[i-j]['close'] - df.iloc[i-j-1]['close']) for j in range(5))
        error_correction_window = close_diff_sum / (current['high'] - current['low']) if (current['high'] - current['low']) != 0 else 0
        
        # Volume Error Correction
        volume_error_signal = current['volume'] / prev1['volume'] - 1 if prev1['volume'] != 0 else 0
        error_correction_quality = abs(current['volume'] - prev1['volume']) / prev2['volume'] if prev2['volume'] != 0 else 0
        correction_persistence = volume_diff_sum2 / current['volume'] if current['volume'] != 0 else 0
        
        # Joint Error Processing
        current_corr = np.sign(current['close'] - prev1['close']) * np.sign(current['volume'] - prev1['volume'])
        prev_corr = np.sign(prev1['close'] - prev2['close']) * np.sign(prev1['volume'] - prev2['volume'])
        correlation_error_detection = abs(current_corr - prev_corr)
        
        error_correction_fidelity = prep_consistency / 5
        
        error_signal_processing = ((current['close'] - current['open']) / np.sqrt(current['volume']) if current['volume'] != 0 else 0) - ((prev1['close'] - prev1['open']) / np.sqrt(prev1['volume']) if prev1['volume'] != 0 else 0)
        
        # Quantum Information Transfer
        # Price Information Flow
        quantum_info_penetration = (current['close'] - prev1['high']) / (prev1['high'] - prev1['low']) if (prev1['high'] - prev1['low']) != 0 else 0
        support_info_transfer = (current['close'] - prev1['low']) / (prev1['high'] - prev1['low']) if (prev1['high'] - prev1['low']) != 0 else 0
        info_transfer_probability = abs(current['close'] - prev1['close']) / (prev1['high'] - prev1['low']) if (prev1['high'] - prev1['low']) != 0 else 0
        
        # Volume Information Flow
        volume_info_rate = current['volume'] / prev1['volume'] if prev1['volume'] != 0 else 0
        info_flow_barrier = (current['volume'] - prev1['volume']) / abs(prev1['volume'] - prev2['volume']) if abs(prev1['volume'] - prev2['volume']) != 0 else 0
        info_transfer_volume = current['volume'] * (current['close'] - prev1['close']) / (current['high'] - current['low']) if (current['high'] - current['low']) != 0 else 0
        
        # Coupled Information Transfer
        synchronized_info_flow = np.sign(current['close'] - prev1['close']) * np.sign(current['volume'] - prev1['volume']) * abs(current['close'] - prev1['close']) / (current['high'] - current['low']) if (current['high'] - current['low']) != 0 else 0
        quantum_transfer_strength = (current['close'] - prev1['close']) * current['volume'] / (current['high'] - current['low']) if (current['high'] - current['low']) != 0 else 0
        
        transfer_persistence = 0
        for j in range(5):
            if i-j-1 >= 0:
                transfer_persistence += np.sign(df.iloc[i-j]['close'] - df.iloc[i-j-1]['close']) * abs(df.iloc[i-j]['close'] - df.iloc[i-j-1]['close']) / (df.iloc[i-j]['high'] - df.iloc[i-j]['low']) if (df.iloc[i-j]['high'] - df.iloc[i-j]['low']) != 0 else 0
        
        # Quantum Alpha Synthesis
        # Combine components with weights
        state_prep_score = (quantum_price_basis + price_state_purity + state_prep_fidelity + 
                           volume_state_init + volume_state_transform + prep_quality + 
                           prep_strength + prep_correlation + prep_consistency) / 9
        
        channel_ops_score = (quantum_price_channel + channel_energy + channel_stability + 
                            volume_channel_capacity + quantum_volume_processing + channel_coherence + 
                            channel_coupling + quantum_channel_interference + channel_resonance) / 9
        
        error_correction_score = (quantum_price_error + error_detection_threshold + error_correction_window + 
                                 volume_error_signal + error_correction_quality + correction_persistence + 
                                 correlation_error_detection + error_correction_fidelity + error_signal_processing) / 9
        
        info_transfer_score = (quantum_info_penetration + support_info_transfer + info_transfer_probability + 
                              volume_info_rate + info_flow_barrier + info_transfer_volume + 
                              synchronized_info_flow + quantum_transfer_strength + transfer_persistence) / 9
        
        # Final Quantum Alpha
        quantum_alpha = (state_prep_score * 0.25 + channel_ops_score * 0.25 + 
                        error_correction_score * 0.25 + info_transfer_score * 0.25)
        
        result.iloc[i] = quantum_alpha
    
    return result
