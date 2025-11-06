import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate basic components
    data['close_prev'] = data['close'].shift(1)
    data['volume_prev'] = data['volume'].shift(1)
    data['volume_2'] = data['volume'].shift(2)
    data['close_5'] = data['close'].shift(5)
    data['volume_3'] = data['volume'].shift(3)
    
    # 1. Regime-Adaptive Gap Momentum
    data['overnight_gap_momentum'] = (data['open'] - data['close_prev']) / data['close_prev']
    data['intraday_gap_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Calculate correlations for multi-frequency gap correlation
    data['abs_close_open'] = abs(data['close'] - data['open'])
    data['high_low_range'] = data['high'] - data['low']
    
    # Short-term correlation (t-2 to t)
    corr_short = []
    for i in range(len(data)):
        if i >= 2:
            window_data = data.iloc[i-2:i+1]
            corr = window_data['abs_close_open'].corr(window_data['high_low_range'])
            corr_short.append(corr if not np.isnan(corr) else 0)
        else:
            corr_short.append(0)
    data['corr_short'] = corr_short
    
    # Medium-term correlation (t-7 to t)
    corr_medium = []
    for i in range(len(data)):
        if i >= 7:
            window_data = data.iloc[i-7:i+1]
            corr = window_data['abs_close_open'].corr(window_data['high_low_range'])
            corr_medium.append(corr if not np.isnan(corr) else 0)
        else:
            corr_medium.append(0)
    data['corr_medium'] = corr_medium
    
    data['multi_freq_gap_corr'] = data['corr_short'] / (data['corr_medium'] + 1e-8)
    data['regime_weighted_gap'] = (data['overnight_gap_momentum'] * (1 + data['multi_freq_gap_corr']) + 
                                  data['intraday_gap_momentum'])
    
    # 2. Volume-Pressure Dynamics
    data['volume_acceleration'] = (data['volume'] / data['volume_prev']) - (data['volume_prev'] / data['volume_2'])
    data['volume_acceleration_prev'] = data['volume_acceleration'].shift(1)
    
    data['volume_regime'] = (np.sign(data['volume_acceleration']) * 
                            (abs(data['volume_acceleration']) / 
                             (abs(data['volume_acceleration']) + abs(data['volume_acceleration_prev']) + 1e-8)))
    
    data['ma_volume_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['ma_volume_20'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['volume_pressure_ratio'] = (data['volume'] / data['ma_volume_5']) / (data['volume'] / data['ma_volume_20'] + 1e-8)
    data['volume_pressure_momentum'] = data['volume_regime'] * data['volume_pressure_ratio']
    
    # 3. Microstructure Signal Quality
    data['true_range_efficiency'] = (data['high'] - data['low']) / (
        abs(data['open'] - data['close_prev']) + abs(data['close'] - data['open']) + abs(data['high'] - data['low']) + 1e-8)
    
    data['gap_noise'] = (abs(data['open'] - data['close_prev']) / (data['close_prev'] + 1e-8) * 
                        (1 - abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)))
    
    data['signal_quality'] = 1 - (data['true_range_efficiency'] / (1 + data['gap_noise']))
    
    # 4. Price-Volume Alignment
    data['short_term_alignment'] = np.sign(data['close'] - data['open']) * np.sign(data['volume'] - data['volume_prev'])
    data['medium_term_alignment'] = np.sign(data['close'] - data['close_5']) * np.sign(data['volume'] - data['ma_volume_5'])
    
    # Calculate alignment consistency
    alignment_consistency = []
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            count = sum(window_data['short_term_alignment'] * window_data['medium_term_alignment'] > 0)
            alignment_consistency.append(count / 5)
        else:
            alignment_consistency.append(0)
    data['alignment_consistency'] = alignment_consistency
    
    data['enhanced_alignment'] = (data['short_term_alignment'] + data['medium_term_alignment']) * (1 + data['alignment_consistency'])
    
    # 5. Regime Transition Detection
    volume_regime_persistence = []
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            count = sum(window_data['volume_regime'] * window_data['volume_regime'].shift(1) > 0)
            volume_regime_persistence.append(count / 5)
        else:
            volume_regime_persistence.append(0)
    data['volume_regime_persistence'] = volume_regime_persistence
    
    data['gap_regime_interaction'] = data['regime_weighted_gap'] * data['volume_regime']
    data['volume_pressure_ratio_3'] = data['volume_pressure_ratio'].shift(3)
    data['pressure_transition'] = (data['volume_pressure_ratio'] - data['volume_pressure_ratio_3']) * data['volume_regime']
    
    data['regime_transition_signal'] = (data['gap_regime_interaction'] * (1 + data['volume_regime_persistence']) + 
                                       data['pressure_transition'])
    
    # 6. Liquidity-Enhanced Gap Dynamics
    data['intraday_liquidity_intensity'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    data['interday_liquidity_persistence'] = data['volume'] / (abs(data['close'] - data['close_prev']) + 1e-8)
    data['liquidity_gap'] = data['intraday_liquidity_intensity'] / (data['interday_liquidity_persistence'] + 1e-8)
    data['liquidity_enhanced_momentum'] = data['regime_transition_signal'] * (1 + data['liquidity_gap'])
    
    # 7. Noise-Adjusted Alpha Synthesis
    data['net_shadow_pressure'] = ((data['high'] - np.maximum(data['open'], data['close']) - 
                                  (np.minimum(data['open'], data['close']) - data['low'])) / 
                                  (data['high'] - data['low'] + 1e-8))
    
    # Calculate ATR for gap decoherence
    data['tr'] = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close_prev']), 
                                     abs(data['low'] - data['close_prev'])))
    data['atr_5'] = data['tr'].rolling(window=5, min_periods=1).mean()
    data['atr_10'] = data['tr'].rolling(window=10, min_periods=1).mean()
    
    # Calculate correlation sign changes
    corr_sign_change = []
    for i in range(len(data)):
        if i >= 1:
            current_sign = np.sign(data['corr_short'].iloc[i])
            prev_sign = np.sign(data['corr_short'].iloc[i-1])
            change = 1 if current_sign != prev_sign else 0
            corr_sign_change.append(change)
        else:
            corr_sign_change.append(0)
    data['corr_sign_change'] = corr_sign_change
    
    data['gap_decoherence'] = (data['corr_sign_change'] * 
                              abs(data['atr_5'] - data['atr_10']) / (data['atr_10'] + 1e-8))
    
    data['alignment_weighted_signal'] = data['enhanced_alignment'] * data['signal_quality']
    
    # Final alpha calculation
    data['final_alpha'] = (data['liquidity_enhanced_momentum'] / 
                          (1 + abs(data['net_shadow_pressure']) + abs(data['gap_decoherence'])) * 
                          (1 + data['alignment_weighted_signal']))
    
    return data['final_alpha']
