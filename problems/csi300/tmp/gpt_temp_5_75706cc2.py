import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Classification
    data['short_term_vol'] = data['close'].rolling(window=5).std()
    data['long_term_vol'] = data['close'].rolling(window=10).std()
    data['volatility_regime'] = data['short_term_vol'] / data['long_term_vol']
    
    # Gap Energy Analysis
    data['gap_magnitude'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Gap Direction Persistence
    gap_sign = np.sign(data['gap_magnitude'])
    gap_persistence = []
    for i in range(len(data)):
        if i < 4:
            gap_persistence.append(0)
        else:
            current_sign = gap_sign.iloc[i]
            window_signs = gap_sign.iloc[i-4:i]
            persistence_count = (window_signs == current_sign).sum()
            gap_persistence.append(persistence_count / 4)
    data['gap_direction_persistence'] = gap_persistence
    
    data['gap_energy'] = data['gap_magnitude'] * np.abs(data['gap_magnitude'])
    
    # Intraday Realization Energy
    data['realization'] = (data['close'] - data['open']) / data['open']
    data['realization_efficiency'] = np.abs(data['realization']) / (np.abs(data['gap_magnitude']) + 0.0001)
    data['realization_energy'] = data['realization'] * np.abs(data['realization'])
    
    # Asymmetric Energy Ratio
    data['upside_energy'] = np.where(data['realization'] > 0, data['realization_energy'], 0)
    data['downside_energy'] = np.where(data['realization'] < 0, data['realization_energy'], 0)
    data['energy_asymmetry'] = (data['upside_energy'] - data['downside_energy']) / (np.abs(data['upside_energy']) + np.abs(data['downside_energy']) + 0.0001)
    
    # Range Efficiency Patterns
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    data['absolute_return'] = np.abs(data['close'] - data['close'].shift(1))
    data['efficiency'] = data['absolute_return'] / (data['true_range'] + 0.0001)
    
    data['upside_efficiency'] = np.where(data['close'] > data['open'], (data['close'] - data['open']) / (data['high'] - data['low'] + 0.0001), 0)
    data['downside_efficiency'] = np.where(data['close'] < data['open'], (data['open'] - data['close']) / (data['high'] - data['low'] + 0.0001), 0)
    data['efficiency_asymmetry'] = (data['upside_efficiency'] - data['downside_efficiency']) / (np.abs(data['upside_efficiency']) + np.abs(data['downside_efficiency']) + 0.0001)
    
    # Volume-Energy Confirmation
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_intensity'] = data['volume'] / (data['volume'].rolling(window=4).mean().shift(1) + 0.0001)
    
    # Volume Consistency
    volume_consistency = []
    for i in range(len(data)):
        if i < 4:
            volume_consistency.append(0)
        else:
            volume_window = data['volume'].iloc[i-4:i+1]
            volume_ma_window = data['volume_ma_5'].iloc[i-4:i+1]
            consistency_count = (volume_window > volume_ma_window).sum()
            volume_consistency.append(consistency_count / 5)
    data['volume_consistency'] = volume_consistency
    
    # Amount Dynamics
    data['amount_ma_5'] = data['amount'].rolling(window=5).mean()
    data['amount_intensity'] = data['amount'] / (data['amount'].rolling(window=4).mean().shift(1) + 0.0001)
    
    # Amount-Volume Correlation
    amount_volume_corr = []
    for i in range(len(data)):
        if i < 4:
            amount_volume_corr.append(0)
        else:
            amount_window = data['amount'].iloc[i-4:i+1]
            volume_window = data['volume'].iloc[i-4:i+1]
            corr = amount_window.corr(volume_window)
            amount_volume_corr.append(corr if not np.isnan(corr) else 0)
    data['amount_volume_correlation'] = amount_volume_corr
    
    data['energy_volume_divergence'] = data['energy_asymmetry'] * (1 - data['volume_consistency'])
    
    # Regime-Adaptive Components
    data['regime_component'] = np.where(
        data['volatility_regime'] > 1.3,
        data['energy_asymmetry'] * data['volume_intensity'],
        np.where(
            data['volatility_regime'] < 0.8,
            data['efficiency_asymmetry'] * data['volume_consistency'],
            data['energy_asymmetry'] * data['efficiency_asymmetry'] * data['gap_direction_persistence']
        )
    )
    
    # Amount-Enhanced Confirmation
    data['large_transaction_indicator'] = np.where(data['amount'] > 2 * data['amount_ma_5'], 1, 0)
    
    data['energy_confirmation'] = np.where(
        (data['large_transaction_indicator'] == 1) & (np.sign(data['energy_asymmetry']) == np.sign(data['gap_magnitude'])),
        1.2,
        np.where(
            (data['large_transaction_indicator'] == 1) & (np.sign(data['energy_asymmetry']) != np.sign(data['gap_magnitude'])),
            0.8,
            1.0
        )
    )
    
    data['efficiency_confirmation'] = np.where(
        (data['amount_volume_correlation'] > 0.7) & (data['volume_consistency'] > 0.6),
        1.3,
        np.where(
            (data['amount_volume_correlation'] < 0.3) | (data['volume_consistency'] < 0.4),
            0.7,
            1.0
        )
    )
    
    # Final Alpha Construction
    data['energy_efficiency_core'] = data['energy_asymmetry'] * data['efficiency_asymmetry']
    data['volume_enhanced_core'] = data['energy_efficiency_core'] * (data['volume_intensity'] + data['amount_intensity']) / 2
    
    # Regime Application
    data['regime_output'] = data['volume_enhanced_core'] * data['regime_component']
    
    # Final Output
    data['amount_weighted_factor'] = data['regime_output'] * data['energy_confirmation'] * data['efficiency_confirmation']
    data['interpretable_alpha'] = data['amount_weighted_factor'] * np.sign(data['gap_magnitude'])
    
    return data['interpretable_alpha']
