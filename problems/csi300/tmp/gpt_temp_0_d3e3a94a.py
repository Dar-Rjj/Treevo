import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # 1. Volatility Regime Assessment
    # True Range calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Average True Range (5-day rolling)
    data['atr_5'] = data['true_range'].rolling(window=5, min_periods=5).mean().shift(1)
    data['volatility_regime'] = data['true_range'] / data['atr_5']
    
    # Regime Duration - count consecutive similar regimes
    data['regime_class'] = np.where(data['volatility_regime'] > 1, 'high', 'low')
    regime_changes = data['regime_class'] != data['regime_class'].shift(1)
    data['regime_group'] = regime_changes.cumsum()
    regime_duration = data.groupby('regime_group').cumcount() + 1
    data['regime_duration'] = regime_duration
    
    # 2. Price Efficiency in Transition
    # Up Move Efficiency
    up_condition = data['close'] > data['open']
    data['up_efficiency'] = np.where(
        up_condition & (data['high'] > data['open']),
        (data['close'] - data['open']) / (data['high'] - data['open']),
        0
    )
    
    # Down Move Efficiency
    down_condition = data['close'] < data['open']
    data['down_efficiency'] = np.where(
        down_condition & (data['open'] > data['low']),
        (data['open'] - data['close']) / (data['open'] - data['low']),
        0
    )
    
    # Directional Efficiency and Transition Efficiency
    data['directional_efficiency'] = data['up_efficiency'] - data['down_efficiency']
    data['transition_efficiency'] = data['directional_efficiency'] * (1 / data['regime_duration'])
    
    # 3. Momentum Quality During Volatility Shifts
    # Regime-Specific Momentum
    data['regime_momentum'] = (data['close'] / data['close'].shift(5)) - 1
    
    # Historical Pattern Matching
    pattern_corr = []
    for i in range(len(data)):
        if i < 25:  # Need enough history for pattern matching
            pattern_corr.append(0)
            continue
            
        current_pattern = data['close'].iloc[i-4:i+1].pct_change().dropna().values
        if len(current_pattern) != 4:
            pattern_corr.append(0)
            continue
            
        correlations = []
        current_regime = data['regime_class'].iloc[i]
        
        for j in range(i-20, i-4):
            if j < 5:
                continue
                
            historical_regime = data['regime_class'].iloc[j]
            if historical_regime != current_regime:
                continue
                
            hist_pattern = data['close'].iloc[j-4:j].pct_change().dropna().values
            if len(hist_pattern) == 4:
                try:
                    corr = np.corrcoef(current_pattern, hist_pattern)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    continue
        
        pattern_corr.append(np.mean(correlations) if correlations else 0)
    
    data['pattern_matching'] = pattern_corr
    data['momentum_consistency'] = data['regime_momentum'] * data['pattern_matching']
    
    # 4. Volume Confirmation in Transition
    # Volume Ratio
    data['avg_volume_4'] = data['volume'].rolling(window=4, min_periods=4).mean()
    data['volume_ratio'] = data['volume'] / data['avg_volume_4']
    
    # Volume Entropy
    data['volume_entropy'] = -data['volume_ratio'] * np.log(data['volume_ratio'] + 1e-8)
    data['transition_volume_score'] = data['volume_entropy'] * data['volume_ratio']
    
    # 5. Final Alpha Construction
    data['core_component'] = data['transition_efficiency'] * data['momentum_consistency']
    data['volume_adjustment'] = data['core_component'] * data['transition_volume_score']
    data['regime_persistence_factor'] = data['volume_adjustment'] * (1 / (1 + data['regime_duration']))
    data['final_alpha'] = data['regime_persistence_factor'] * data['true_range']
    
    # Return the final alpha factor series
    return data['final_alpha']
