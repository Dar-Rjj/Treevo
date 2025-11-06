import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Helper functions
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    # Volatility-Behavioral Divergence Framework
    # Gap Efficiency Divergence
    data['short_term_gap_eff'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['medium_term_gap_eff'] = np.abs(data['close'] - data['open'].shift(5)) / (
        (data['high'] - data['low']).rolling(window=6).sum() + 1e-8)
    data['gap_eff_divergence'] = data['short_term_gap_eff'] - data['medium_term_gap_eff']
    
    # Volatility Asymmetry Analysis
    data['upper_vol_asymmetry'] = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['lower_vol_asymmetry'] = (data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['vol_asymmetry'] = data['upper_vol_asymmetry'] - data['lower_vol_asymmetry']
    
    # Behavioral Pressure System
    data['morning_gap_pressure'] = (data['high'] - data['open']) / (np.abs(data['open'] - data['close'].shift(1)) + 1e-8)
    data['gap_fill_pressure'] = (data['close'] - data['open']) / (np.abs(data['open'] - data['close'].shift(1)) + 1e-8)
    data['pressure_asymmetry'] = data['morning_gap_pressure'] - data['gap_fill_pressure']
    
    # Combine Volatility-Behavioral Divergence
    data['vol_behavioral_div'] = data['gap_eff_divergence'] * data['vol_asymmetry'] * data['pressure_asymmetry']
    
    # Entropy-Momentum Dynamics
    # Multi-Scale Momentum
    data['short_term_momentum'] = (data['close'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(5)) / (data['high'].shift(5) - data['low'].shift(5) + 1e-8)
    data['momentum_reversal_ratio'] = data['short_term_momentum'] / (data['medium_term_momentum'] + 1e-8)
    
    # Behavioral Entropy Components
    data['price_entropy'] = np.sign(data['close'] - data['close'].shift(1)) * np.log(np.abs(data['close'] - data['close'].shift(1)) + 1)
    data['behavioral_entropy'] = -np.abs(data['close'] - data['open']) * np.log(np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8) + 1e-8)
    data['volume_behavioral_entanglement'] = (np.sign(data['close'] - data['open']) * 
                                             np.sign(data['volume'] - data['volume'].shift(1)) * 
                                             (data['volume'] - data['volume'].shift(1)) / (data['volume'].shift(1) + 1e-8))
    
    # Combine Entropy-Momentum
    data['entropy_momentum'] = (data['momentum_reversal_ratio'] * data['price_entropy'] * 
                               data['behavioral_entropy'] * data['volume_behavioral_entanglement'])
    
    # Volume-Pressure Confirmation
    # Volume Asymmetry Analysis
    returns = data['close'].pct_change()
    up_days = returns > 0
    data['upside_volume_ratio'] = (data['volume'].rolling(window=10).apply(
        lambda x: x[up_days.loc[x.index]].mean() if up_days.loc[x.index].any() else 0) / 
        data['volume'].rolling(window=10).mean())
    
    pos_returns = np.maximum(returns, 0)
    neg_returns = np.maximum(-returns, 0)
    data['price_asymmetry'] = (np.log(1 + pos_returns.rolling(window=10).sum()) - 
                              np.log(1 + neg_returns.rolling(window=10).sum()))
    data['volume_asymmetry'] = data['upside_volume_ratio'] * data['price_asymmetry']
    
    # Volume Cluster Dynamics
    data['gap_turnover_momentum'] = (data['volume'] * data['close']) / (
        (data['volume'] * data['close']).shift(1).rolling(window=4).max() + 1e-8)
    data['volume_persistence'] = data['volume'] / (data['volume'].shift(2) + 1e-8)
    data['volume_stress'] = data['volume'] / (data['volume'].rolling(window=5).mean() + 1e-8)
    
    # Combine Volume-Pressure
    data['volume_pressure'] = data['volume_asymmetry'] * data['gap_turnover_momentum'] * data['volume_persistence'] / (data['volume_stress'] + 1e-8)
    
    # Breakout Enhancement System
    # Volatility Break Detection
    data['gap_vol_compression'] = (np.abs(data['close'] - data['open']).rolling(window=5).sum() / 
                                  (np.abs(data['close'] - data['open']).rolling(window=10).sum() + 1e-8) - 1)
    
    data['true_range_vol'] = true_range(data['high'], data['low'], data['close'].shift(1)).rolling(window=20).mean()
    data['high_low_range_expansion'] = (data['high'].rolling(window=20).max() / 
                                       (data['low'].rolling(window=20).min() + 1e-8) - 1)
    
    # Behavioral Break Detection
    data['behavioral_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['behavioral_break'] = data['behavioral_efficiency'] / (data['behavioral_efficiency'].shift(1) + 1e-8)
    data['momentum_break'] = ((data['close'] - data['high'].rolling(window=5).max()) / 
                             (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min() + 1e-8))
    
    # Combine Breakout Enhancement
    data['breakout_enhancement'] = (data['momentum_break'] * (1 + np.abs(data['behavioral_break'])) * 
                                   data['high_low_range_expansion'] / (data['true_range_vol'] + 1e-8))
    
    # Market State Classification
    data['volatility_state'] = np.where(data['high'] - data['low'] > data['high'].shift(5) - data['low'].shift(5), 
                                       'high', 
                                       np.where(data['high'] - data['low'] < data['high'].shift(5) - data['low'].shift(5), 
                                               'low', 'stable'))
    
    data['efficiency_state'] = np.where(data['behavioral_efficiency'] > 0.6, 'high',
                                       np.where(data['behavioral_efficiency'] < 0.4, 'low', 'medium'))
    
    # Base Component Selection
    conditions = [
        (data['volatility_state'] == 'high') & (data['efficiency_state'] == 'high'),
        (data['volatility_state'] == 'low') & (data['efficiency_state'] == 'low'),
        (data['volatility_state'] == 'stable') & (data['efficiency_state'] == 'medium')
    ]
    choices = [
        data['vol_behavioral_div'],
        data['volume_pressure'],
        data['entropy_momentum']
    ]
    data['base_component'] = np.select(conditions, choices, 
                                      default=(data['vol_behavioral_div'] + data['volume_pressure'] + data['entropy_momentum']) / 3)
    
    # Gap-Behavioral Adjustment
    data['gap_absorption'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    data['upper_rejection'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['lower_rejection'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['rejection_balance'] = data['upper_rejection'] - data['lower_rejection']
    data['gap_adjustment'] = (1 - np.abs(data['rejection_balance'])) * data['gap_absorption']
    
    # Final Alpha Synthesis
    data['enhanced_signal'] = data['base_component'] * data['breakout_enhancement']
    data['gap_adjusted_signal'] = data['enhanced_signal'] * data['gap_adjustment']
    data['final_alpha'] = data['gap_adjusted_signal'] * data['gap_vol_compression']
    
    return data['final_alpha']
