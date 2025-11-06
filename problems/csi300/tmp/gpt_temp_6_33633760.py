import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    
    # Multi-day shifts
    for shift_val in [2, 3, 4, 5, 6]:
        data[f'close_t_minus_{shift_val}'] = data['close'].shift(shift_val)
        data[f'volume_t_minus_{shift_val}'] = data['volume'].shift(shift_val)
        data[f'high_t_minus_{shift_val}'] = data['high'].shift(shift_val)
        data[f'low_t_minus_{shift_val}'] = data['low'].shift(shift_val)
    
    # Calculate rolling volume average
    data['volume_6d_avg'] = data['volume'].shift(1).rolling(window=6, min_periods=1).mean()
    
    # Intraday Wave Asymmetry
    data['opening_wave'] = ((data['open'] - data['prev_close']) * 
                           (data['close'] - data['open']) / 
                           (data['high'] - data['low'] + 1e-8) * 
                           data['volume'] / (data['prev_volume'] + 1e-8))
    
    data['midday_wave'] = (((data['high'] + data['low']) / 2 - (data['open'] + data['close']) / 2) * 
                          data['volume'] / (data['prev_volume'] + 1e-8) * 
                          np.sign(data['close'] - data['prev_close']))
    
    data['closing_wave'] = ((data['close'] - data['open']) * 
                           np.abs(data['close'] - (data['high'] + data['low']) / 2) / 
                           (data['high'] - data['low'] + 1e-8) * 
                           data['volume'] / (data['prev_volume'] + 1e-8))
    
    # Multi-Day Wave Asymmetry
    data['three_day_wave'] = ((data['close'] - data['close_t_minus_2']) * 
                             data['volume'] / (data['volume_t_minus_2'] + 1e-8) * 
                             (data['high'] - data['low']) / (data['prev_high'] - data['prev_low'] + 1e-8))
    
    data['five_day_wave'] = (np.sign(data['close'] - data['close_t_minus_4']) * 
                            np.sign(data['volume'] - data['volume_t_minus_4']) * 
                            (data['high'] - data['low']) / (data['high_t_minus_3'] - data['low_t_minus_3'] + 1e-8))
    
    data['weekly_wave'] = ((data['close'] - data['close_t_minus_6']) * 
                          data['volume'] / (data['volume_6d_avg'] + 1e-8) * 
                          (data['high'] - data['low']) / (data['high_t_minus_5'] - data['low_t_minus_5'] + 1e-8))
    
    # Frequency Wave Asymmetry
    data['high_freq_wave'] = ((data['close'] - data['prev_close']) * 
                             data['volume'] / (data['prev_volume'] + 1e-8) * 
                             (data['high'] - data['low']) / (data['prev_high'] - data['prev_low'] + 1e-8))
    
    data['low_freq_wave'] = ((data['close'] - data['close_t_minus_3']) * 
                            data['volume'] / (data['volume_t_minus_3'] + 1e-8) * 
                            (data['high'] - data['low']) / (data['high_t_minus_3'] - data['low_t_minus_3'] + 1e-8))
    
    data['mixed_freq_wave'] = data['high_freq_wave'] * data['low_freq_wave']
    
    # Wave Volume-Price Asymmetry
    data['vwap_t'] = data['amount'] / (data['volume'] + 1e-8)
    data['prev_vwap'] = data['prev_amount'] / (data['prev_volume'] + 1e-8)
    
    data['price_volume_alignment'] = (np.sign(data['close'] - data['prev_close']) * 
                                     np.sign(data['volume'] - data['prev_volume']) * 
                                     (data['high'] - data['low']) * 
                                     data['vwap_t'] / (data['prev_vwap'] + 1e-8))
    
    data['opening_volume'] = (np.sign(data['open'] - data['prev_close']) * 
                             np.sign(data['volume'] - data['prev_volume']) * 
                             np.abs(data['open'] - data['prev_close']) * 
                             data['volume'] / (data['volume'].shift(2) + 1e-8))
    
    # Wave Asymmetry Divergence
    data['price_volume_break'] = (data['price_volume_alignment'].rolling(window=5, min_periods=1)
                                 .apply(lambda x: (x < 0).sum()) * 
                                 data['volume'] * 
                                 (data['high'] - data['low']) / (data['prev_high'] - data['prev_low'] + 1e-8))
    
    divergence_mask = np.sign(data['close'] - data['open']) != np.sign(data['volume'] - data['prev_volume'])
    data['closing_divergence'] = ((data['close'] - data['open']) * 
                                 data['volume'] / (data['prev_volume'] + 1e-8) * 
                                 (data['high'] - data['low']) / (data['prev_high'] - data['prev_low'] + 1e-8))
    data['closing_divergence'] = data['closing_divergence'].where(divergence_mask, 0)
    
    # Wave Transition Signals
    volume_spike_mask = data['volume'] > 2 * data['prev_volume']
    data['volume_spike'] = ((data['close'] - data['open']) * 
                           (data['high'] - data['low']) / (data['prev_high'] - data['prev_low'] + 1e-8))
    data['volume_spike'] = data['volume_spike'].where(volume_spike_mask, 0)
    
    low_volume_mask = data['volume'] < 0.6 * data['prev_volume']
    data['low_volume'] = ((data['close'] - (data['high'] + data['low']) / 2) * 
                         np.abs(data['high'] - data['low']) / (data['prev_high'] - data['prev_low'] + 1e-8))
    data['low_volume'] = data['low_volume'].where(low_volume_mask, 0)
    
    # Wave Amplitude Asymmetry
    data['high_low_asymmetry'] = ((data['high'] - data['low']) * 
                                 (data['close'] - data['open']) / (np.abs(data['close'] - data['open']) + 1e-8) * 
                                 (data['high'] - data['low']) / (data['prev_high'] - data['prev_low'] + 1e-8))
    
    data['closing_amplitude'] = (np.abs(data['close'] - (data['high'] + data['low']) / 2) * 
                                (data['high'] - data['low']) * 
                                data['volume'] * 
                                (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8))
    
    # Wave Amplitude Memory
    data['amplitude_persistence'] = ((data['high'] - data['low']) / (data['prev_high'] - data['prev_low'] + 1e-8) * 
                                    data['volume'] / (data['prev_volume'] + 1e-8) * 
                                    (data['high'] - data['low']) / (data['high_t_minus_3'] - data['low_t_minus_3'] + 1e-8))
    
    breakout_mask = (data['high'] - data['low']) > 1.3 * (data['prev_high'] - data['prev_low'])
    data['amplitude_breakout'] = ((data['high'] - data['low']) * 
                                 data['volume'] * 
                                 (data['high'] - data['low']) / (data['high_t_minus_3'] - data['low_t_minus_3'] + 1e-8))
    data['amplitude_breakout'] = data['amplitude_breakout'].where(breakout_mask, 0)
    
    # Multi-Scale Wave Amplitude
    data['short_term_amplitude'] = ((data['high'] - data['low']) / (data['high_t_minus_2'] - data['low_t_minus_2'] + 1e-8) * 
                                   data['volume'] / (data['volume_t_minus_2'] + 1e-8) * 
                                   (data['high'] - data['low']) / (data['prev_high'] - data['prev_low'] + 1e-8))
    
    data['medium_term_amplitude'] = ((data['high'] - data['low']) / (data['high_t_minus_5'] - data['low_t_minus_5'] + 1e-8) * 
                                    data['volume'] / (data['volume_t_minus_5'] + 1e-8) * 
                                    (data['high'] - data['low']) / (data['high_t_minus_3'] - data['low_t_minus_3'] + 1e-8))
    
    # Wave Momentum Efficiency
    data['gap_momentum'] = ((data['close'] - data['open']) / (np.abs(data['open'] - data['prev_close']) + 1e-8) * 
                           np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8) * 
                           (data['close'] - data['prev_close']))
    
    data['opening_pressure'] = ((data['open'] - data['prev_close']) * 
                               data['volume'] / (data['amount'] + 1e-8) * 
                               (data['close'] - data['prev_close']))
    
    data['closing_transition'] = (np.abs(data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'] + 1e-8) * 
                                 data['volume'] / (data['prev_volume'] + 1e-8) * 
                                 (data['close'] - data['prev_close']))
    
    # Core Wave Signals
    data['temporal_core'] = data['opening_wave'] * data['three_day_wave']
    data['phase_core'] = data['price_volume_alignment'] * data['volume_spike']
    data['amplitude_core'] = data['high_low_asymmetry'] * data['amplitude_breakout']
    
    # Multi-Frequency Integration
    data['high_freq_composite'] = data['high_freq_wave'] * data['price_volume_alignment']
    data['medium_freq_composite'] = data['five_day_wave'] * data['medium_term_amplitude']
    data['cross_freq_composite'] = data['high_freq_composite'] * data['low_freq_wave']
    
    # Final Alpha Construction
    data['temporal_alpha'] = data['temporal_core'] * data['cross_freq_composite']
    data['phase_alpha'] = data['phase_core'] * (data['price_volume_alignment'] + data['opening_volume'])
    data['integrated_alpha'] = (data['temporal_alpha'] * data['phase_alpha'] * 
                               data['amplitude_core'] * 
                               (data['gap_momentum'] + data['opening_pressure'] + data['closing_transition']))
    
    # Market Regime Adaptation
    high_vol_mask = (data['high'] - data['low']) > 1.4 * (data['high_t_minus_5'] - data['low_t_minus_5'])
    low_vol_mask = (data['high'] - data['low']) < 0.75 * (data['high_t_minus_5'] - data['low_t_minus_5'])
    
    data['regime_weight'] = 1.0
    data.loc[high_vol_mask, 'regime_weight'] = 1.2
    data.loc[low_vol_mask, 'regime_weight'] = 0.8
    
    # Volume Regime Adjustment
    high_vol_mask = data['volume'] > 1.6 * data['prev_volume']
    low_vol_mask = data['volume'] < 0.65 * data['prev_volume']
    
    data['volume_weight'] = 1.0
    data.loc[high_vol_mask, 'volume_weight'] = 1.15
    data.loc[low_vol_mask, 'volume_weight'] = 0.85
    
    # Final weighted alpha factor
    data['final_alpha'] = (data['integrated_alpha'] * 
                          data['regime_weight'] * 
                          data['volume_weight'] * 
                          (data['high'] - data['low']) / (data['prev_high'] - data['prev_low'] + 1e-8) * 
                          data['volume'] / (data['prev_volume'] + 1e-8))
    
    # Clean up and return
    alpha_series = data['final_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
