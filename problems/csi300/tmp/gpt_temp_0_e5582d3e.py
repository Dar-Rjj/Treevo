import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Price Momentum Dynamics
    data['raw_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['gap_momentum'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['open'] + 1e-8)
    
    # Momentum Acceleration
    data['momentum_change_rate'] = (data['close']/data['close'].shift(1) - 1) / (data['close'].shift(1)/data['close'].shift(2) - 1 + 1e-8)
    data['momentum_persistence'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['close'].shift(1) - data['close'].shift(2))
    
    # Asymmetric Momentum
    data['upside_momentum'] = (data['high'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['downside_momentum'] = (data['close'].shift(1) - data['low']) / (data['close'].shift(1) + 1e-8)
    data['momentum_efficiency'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    # Volume Contrarian Signals
    data['volume_spike'] = data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3 + 1e-8) - 1
    data['volume_reversal'] = np.sign(data['volume'] - data['volume'].shift(1)) * np.sign(data['close'] - data['close'].shift(1))
    data['volume_divergence'] = data['volume'] / data['volume'].shift(1) - (data['close'] / data['close'].shift(1) - 1)
    
    # Contrarian Volume Patterns
    data['high_volume_pullback'] = (data['close'] < data['close'].shift(1)) * (data['volume'] > data['volume'].shift(1))
    data['low_volume_breakout'] = (data['close'] > data['close'].shift(1)) * (data['volume'] < data['volume'].shift(1))
    data['volume_exhaustion'] = data['volume'] / data['volume'].shift(1) * (data['close'] / data['close'].shift(1) - 1)
    
    # Smart Money Detection
    data['large_trade_concentration'] = data['amount'] / (data['volume'] * data['close'] + 1e-8)
    
    vwap_t = data['amount'] / (data['volume'] + 1e-8)
    vwap_t_1 = data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8)
    vwap_t_2 = data['amount'].shift(2) / (data['volume'].shift(2) + 1e-8)
    vwap_t_3 = data['amount'].shift(3) / (data['volume'].shift(3) + 1e-8)
    
    data['institutional_flow'] = vwap_t / ((vwap_t_1 + vwap_t_2 + vwap_t_3) / 3 + 1e-8) - 1
    data['smart_volume_alignment'] = np.sign(vwap_t - vwap_t_1) * np.sign(data['close'] - data['close'].shift(1))
    
    # Regime Switching Logic
    data['strong_momentum_regime'] = ((data['close']/data['close'].shift(1) - 1 > data['close'].shift(1)/data['close'].shift(2) - 1) & 
                                    (data['close'].shift(1)/data['close'].shift(2) - 1 > data['close'].shift(2)/data['close'].shift(3) - 1)).astype(int)
    
    data['momentum_reversal_regime'] = ((data['close']/data['close'].shift(1) - 1 < 0) & 
                                      (data['close'].shift(1)/data['close'].shift(2) - 1 > 0)).astype(int)
    
    data['consolidation_regime'] = ((abs(data['close']/data['close'].shift(1) - 1) < abs(data['close'].shift(1)/data['close'].shift(2) - 1)) & 
                                  (abs(data['close'].shift(1)/data['close'].shift(2) - 1) < abs(data['close'].shift(2)/data['close'].shift(3) - 1))).astype(int)
    
    # Volume Regime Classification
    data['high_volume_regime'] = ((data['volume'] > data['volume'].shift(1)) & 
                                (data['volume'].shift(1) > data['volume'].shift(2))).astype(int)
    
    data['low_volume_regime'] = ((data['volume'] < data['volume'].shift(1)) & 
                               (data['volume'].shift(1) < data['volume'].shift(2))).astype(int)
    
    data['volume_normalization'] = data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3 + 1e-8)
    
    # Regime Transition Triggers
    data['momentum_breakout_signal'] = data['strong_momentum_regime'] * data['low_volume_breakout']
    data['contrarian_reversal_signal'] = data['momentum_reversal_regime'] * data['high_volume_pullback']
    data['smart_money_confirmation'] = data['consolidation_regime'] * data['smart_volume_alignment']
    
    # Price Efficiency Anomalies
    data['overreaction_measure'] = abs(data['close']/data['close'].shift(1) - 1) / (abs(data['high'] - data['low']) / data['close'].shift(1) + 1e-8)
    data['momentum_decay'] = (data['close']/data['close'].shift(1) - 1) / (data['close'].shift(1)/data['close'].shift(2) - 1 + 1e-8)
    data['price_slippage'] = (data['high'] - data['close']) / (data['close'] - data['low'] + 1e-8)
    
    data['volume_impact_ratio'] = abs(data['close'] - data['close'].shift(1)) / (data['volume'] + 1e-8)
    data['volume_persistence'] = data['volume'] / data['volume'].shift(1) * data['volume'].shift(1) / data['volume'].shift(2)
    data['efficient_volume'] = (data['close'] - data['close'].shift(1)) * data['volume'] / (data['high'] - data['low'] + 1e-8)
    
    # Multi-timeframe Confirmation
    data['multi_period_momentum'] = (data['close']/data['close'].shift(3) - 1) / (data['close'].shift(3)/data['close'].shift(6) - 1 + 1e-8)
    
    # Calculate rolling metrics
    for i in range(len(data)):
        if i >= 2:
            # Momentum Stability
            momentum_signs = [np.sign(data['close'].iloc[j] - data['close'].iloc[j-1]) for j in range(i-2, i+1)]
            data.loc[data.index[i], 'momentum_stability'] = sum(1 for sign in momentum_signs if sign > 0) / 3
            
            # Volume Regime Stability
            volume_increases = [1 if data['volume'].iloc[j] > data['volume'].iloc[j-1] else 0 for j in range(i-2, i+1)]
            data.loc[data.index[i], 'volume_regime_stability'] = sum(volume_increases) / 3
            
            # Volume-Momentum Correlation
            correlations = [np.sign(data['volume'].iloc[j] - data['volume'].iloc[j-1]) * np.sign(data['close'].iloc[j] - data['close'].iloc[j-1]) 
                          for j in range(i-2, i+1)]
            data.loc[data.index[i], 'volume_momentum_correlation'] = sum(correlations) / 3
            
            # Smart Money Consistency
            smart_alignments = [data['smart_volume_alignment'].iloc[j] for j in range(i-2, i+1)]
            data.loc[data.index[i], 'smart_money_consistency'] = sum(1 for align in smart_alignments if align > 0) / 3
    
    # Fill NaN values with 0
    data = data.fillna(0)
    
    # Regime Transition Strength
    data['momentum_transition_power'] = data['momentum_change_rate'] * data['momentum_persistence']
    data['volume_transition_power'] = data['volume_spike'] * data['volume_reversal']
    data['regime_alignment_strength'] = data['smart_money_confirmation'] * data['momentum_breakout_signal']
    
    # Alpha Construction
    data['momentum_breakout_alpha'] = data['momentum_breakout_signal'] * data['momentum_transition_power']
    data['contrarian_reversal_alpha'] = data['contrarian_reversal_signal'] * data['volume_transition_power']
    data['smart_money_alpha'] = data['smart_money_confirmation'] * data['regime_alignment_strength']
    
    # Regime-Adaptive Weighting
    data['momentum_weighted_signals'] = data['momentum_breakout_alpha'] * (1 + data['raw_momentum'])
    data['volume_weighted_signals'] = data['contrarian_reversal_alpha'] * (1 + data['volume_spike'])
    data['efficiency_weighted_signals'] = data['smart_money_alpha'] * (1 + data['momentum_efficiency'])
    
    # Final Alpha Components
    data['primary_alpha'] = data['momentum_weighted_signals'] * data['momentum_stability']
    data['secondary_alpha'] = data['volume_weighted_signals'] * data['volume_momentum_correlation']
    
    # Composite Alpha
    alpha = data['primary_alpha'] + data['secondary_alpha'] + data['efficiency_weighted_signals']
    
    return alpha
