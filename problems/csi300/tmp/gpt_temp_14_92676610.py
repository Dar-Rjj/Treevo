import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Gap-Pressure Divergence Analysis
    # Gap Efficiency Components
    data['short_term_gap_eff'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['medium_term_gap_eff'] = np.abs(data['close'] - data['open'].shift(5)) / (
        (data['high'] - data['low']).rolling(window=6).sum()
    )
    data['gap_eff_divergence'] = data['short_term_gap_eff'] - data['medium_term_gap_eff']
    
    # Pressure Asymmetry Components
    data['morning_gap_pressure'] = (data['high'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    data['gap_fill_pressure'] = (data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    data['pressure_asymmetry'] = data['morning_gap_pressure'] - data['gap_fill_pressure']
    
    # Gap-Pressure Integration
    data['raw_gap_pressure'] = data['gap_eff_divergence'] * data['pressure_asymmetry']
    data['enhanced_gap_pressure'] = data['raw_gap_pressure'] * np.sign(data['close'] - data['open'])
    
    # Microstructure Efficiency System
    # Efficiency Measurement
    data['price_movement_eff'] = ((data['close'] - data['open']) / (data['high'] - data['low'])) * np.sign(data['close'] - data['open'])
    data['volume_eff'] = (data['volume'] / data['volume'].rolling(window=5).mean()) * data['price_movement_eff']
    data['gap_efficiency'] = ((data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))) * (data['volume'] / data['volume'].shift(1))
    
    # Efficiency Regime Detection
    data['efficiency_trend'] = data['price_movement_eff'] - data['price_movement_eff'].rolling(window=3).mean()
    data['volume_confirmation'] = data['volume_eff'] * data['gap_efficiency']
    data['efficiency_regime'] = data['efficiency_trend'] * data['volume_confirmation']
    
    # Weighted Efficiency Construction
    data['raw_efficiency'] = data['price_movement_eff'] * data['efficiency_regime']
    data['volume_weighted_eff'] = data['raw_efficiency'] * data['volume_eff']
    
    # Volume-Pressure Confirmation
    # Pressure Accumulation Patterns
    data['buy_sell_pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low'])) - ((data['high'] - data['close']) / (data['high'] - data['low']))
    
    def count_positive_changes(series):
        return (series > series.shift(1)).rolling(window=4).sum()
    
    data['pressure_persistence'] = count_positive_changes(data['buy_sell_pressure'])
    data['pressure_reversal'] = np.abs(data['buy_sell_pressure'] - data['buy_sell_pressure'].shift(1)) / (data['high'] - data['low'])
    
    # Volume Asymmetry Components
    def avg_volume_on_up_days(window_data):
        up_days = window_data[data['close'] > data['open']]
        return up_days.mean() if len(up_days) > 0 else 0
    
    data['upside_volume_ratio'] = data['volume'].rolling(window=10).apply(
        lambda x: avg_volume_on_up_days(x) / x.mean() if x.mean() > 0 else 0, raw=False
    )
    
    def price_asymmetry_calc(window_data):
        returns = window_data.pct_change().dropna()
        pos_returns = returns[returns > 0].sum()
        neg_returns = returns[returns < 0].sum()
        return np.log(1 + pos_returns) - np.log(1 + abs(neg_returns))
    
    data['price_asymmetry'] = data['close'].rolling(window=10).apply(price_asymmetry_calc, raw=False)
    data['volume_asymmetry'] = data['upside_volume_ratio'] * data['price_asymmetry']
    
    # Volume-Pressure Integration
    data['raw_volume_pressure'] = data['pressure_persistence'] * data['volume_asymmetry']
    data['enhanced_volume_pressure'] = data['raw_volume_pressure'] * data['pressure_reversal']
    
    # Convergence Momentum System
    # Gap-Efficiency Alignment
    data['gap_efficiency_correlation'] = data['enhanced_gap_pressure'] * data['volume_weighted_eff']
    data['pressure_efficiency_alignment'] = data['enhanced_volume_pressure'] * data['volume_weighted_eff']
    data['convergence_strength'] = data['gap_efficiency_correlation'] * data['pressure_efficiency_alignment']
    
    # Momentum Enhancement
    data['base_convergence'] = data['convergence_strength'] * data['pressure_persistence']
    data['enhanced_convergence'] = data['base_convergence'] * data['pressure_reversal']
    
    # Adaptive Regime Filtering
    # Regime Signal Construction
    data['core_microstructure_signal'] = data['enhanced_gap_pressure'] * data['volume_weighted_eff']
    data['pressure_enhanced_signal'] = data['core_microstructure_signal'] * data['enhanced_volume_pressure']
    
    # Dynamic Regime Weighting
    data['high_activity_weight'] = np.abs(data['buy_sell_pressure']) * (data['volume'] / data['volume'].shift(1))
    data['low_activity_weight'] = data['gap_efficiency'] * data['pressure_persistence']
    data['adaptive_microstructure_weight'] = data['high_activity_weight'] / (data['high_activity_weight'] + data['low_activity_weight'])
    
    # Final Signal Integration
    data['raw_microstructure_alpha'] = data['pressure_enhanced_signal'] * data['adaptive_microstructure_weight']
    data['convergence_filtered_alpha'] = data['raw_microstructure_alpha'] * data['enhanced_convergence']
    
    # Alpha Output Generation
    data['primary_alpha_signal'] = data['convergence_filtered_alpha'] * data['efficiency_regime']
    data['final_alpha'] = data['primary_alpha_signal'] * data['volume_asymmetry']
    
    return data['final_alpha']
