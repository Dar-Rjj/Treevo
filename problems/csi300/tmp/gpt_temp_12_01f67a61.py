import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Scale Gap Efficiency Patterns
    df['short_term_gap_eff'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'])
    df['medium_term_gap_eff'] = np.abs(df['close'] - df['open'].shift(3)) / (df['high'].shift(3) - df['low'].shift(3))
    df['fractal_gap_gradient'] = df['short_term_gap_eff'] / df['medium_term_gap_eff']
    
    # Momentum Pressure with Gap Analysis
    df['upward_gap_pressure'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['lower_shadow_eff'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['gap_pressure_asymmetry'] = df['upward_gap_pressure'] - df['lower_shadow_eff']
    
    # Multi-Timeframe Gap Decay
    df['range_sum_2d'] = (df['high'] - df['low']).rolling(window=2).sum()
    df['range_sum_5d'] = (df['high'] - df['low']).rolling(window=5).sum()
    df['gap_eff_2d'] = np.abs(df['close'] - df['close'].shift(1)) / df['range_sum_2d']
    df['gap_eff_5d'] = np.abs(df['close'] - df['close'].shift(4)) / df['range_sum_5d']
    df['gap_decay_divergence'] = df['gap_eff_2d'] - df['gap_eff_5d']
    
    # Volume-Gap Compression Dynamics
    df['daily_turnover'] = df['volume'] * df['close']
    df['gap_flow_compression'] = (df['amount'] - df['amount'].shift(1)) / df['amount'].shift(1)
    df['volume_gap_eff'] = np.abs(df['gap_flow_compression']) / (df['high'] - df['low'])
    
    # Pressure Resonance Framework
    df['morning_gap_pressure'] = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'])
    df['gap_fill_pressure'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['high_freq_pressure'] = df['morning_gap_pressure'] - df['gap_fill_pressure']
    
    df['pressure_asymmetry_ma'] = df['gap_pressure_asymmetry'].rolling(window=6).mean()
    
    # Calculate rolling correlation for pressure resonance
    pressure_resonance = []
    for i in range(len(df)):
        if i >= 7:
            window_high_freq = df['high_freq_pressure'].iloc[i-7:i+1]
            window_medium_freq = df['pressure_asymmetry_ma'].iloc[i-7:i+1]
            if len(window_high_freq.dropna()) >= 5 and len(window_medium_freq.dropna()) >= 5:
                corr = window_high_freq.corr(window_medium_freq)
                pressure_resonance.append(corr if not np.isnan(corr) else 0)
            else:
                pressure_resonance.append(0)
        else:
            pressure_resonance.append(0)
    df['pressure_resonance'] = pressure_resonance
    
    # Fractal Regime Gap Components
    df['high_efficiency_factor'] = df['fractal_gap_gradient'] * df['gap_pressure_asymmetry'] * df['volume_gap_eff']
    df['low_efficiency_factor'] = df['gap_decay_divergence'] * np.abs(df['gap_pressure_asymmetry'] - 0.5) * (df['volume'] / df['volume'].shift(1))
    df['transition_factor'] = df['volume_gap_eff'] * (df['close'] / df['close'].shift(1) - 1) * df['gap_flow_compression']
    
    # Behavioral Gap Transitions
    df['gap_microstructure'] = np.where(df['fractal_gap_gradient'] > 1.2, 'high',
                                       np.where(df['fractal_gap_gradient'] < 0.8, 'low', 'normal'))
    df['transition_momentum'] = df['fractal_gap_gradient'] - df['fractal_gap_gradient'].shift(3)
    df['early_warning_signal'] = df['transition_momentum'] * np.abs(df['gap_decay_divergence'])
    
    # Cross-Dimensional Gap Information
    gap_days = np.abs(df['close'] - df['open']) / (df['high'] - df['low']) > 0.3
    
    # Calculate rolling correlations for gap information flow
    price_volume_gap_lead = []
    volume_amount_gap_lead = []
    
    for i in range(len(df)):
        if i >= 5 and gap_days.iloc[i]:
            # Price-volume gap lead
            price_changes = (df['close'] - df['open']).iloc[max(0, i-4):i+1]
            next_volumes = df['volume'].iloc[max(0, i-3):min(len(df), i+2)]
            if len(price_changes.dropna()) >= 3 and len(next_volumes.dropna()) >= 3:
                corr1 = price_changes.corr(next_volumes.iloc[:len(price_changes)])
                price_volume_gap_lead.append(corr1 if not np.isnan(corr1) else 0)
            else:
                price_volume_gap_lead.append(0)
            
            # Volume-amount gap lead
            volumes = df['volume'].iloc[max(0, i-4):i+1]
            next_amounts = df['amount'].iloc[max(0, i-3):min(len(df), i+2)]
            if len(volumes.dropna()) >= 3 and len(next_amounts.dropna()) >= 3:
                corr2 = volumes.corr(next_amounts.iloc[:len(volumes)])
                volume_amount_gap_lead.append(corr2 if not np.isnan(corr2) else 0)
            else:
                volume_amount_gap_lead.append(0)
        else:
            price_volume_gap_lead.append(0)
            volume_amount_gap_lead.append(0)
    
    df['price_volume_gap_lead'] = price_volume_gap_lead
    df['volume_amount_gap_lead'] = volume_amount_gap_lead
    df['gap_information_flow'] = df['price_volume_gap_lead'] * df['volume_amount_gap_lead']
    
    # Fractal Gap Alpha Synthesis
    df['price_gap_divergence'] = np.abs((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) - np.abs(df['gap_decay_divergence'])
    df['efficiency_flow_mismatch'] = df['fractal_gap_gradient'] - df['volume_gap_eff']
    
    # Regime-adaptive weighting
    regime_weights = np.where(df['gap_microstructure'] == 'high', 1.2,
                             np.where(df['gap_microstructure'] == 'low', 0.8, 1.0))
    
    # Final alpha signal
    alpha_signal = (df['price_gap_divergence'] * 
                   df['efficiency_flow_mismatch'] * 
                   df['gap_information_flow'] * 
                   (df['volume'] / df['volume'].shift(1)) * 
                   regime_weights * 
                   df['pressure_resonance'] * 
                   (1 - np.abs(df['early_warning_signal'])))
    
    return alpha_signal
