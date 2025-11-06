import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Price Structure Analysis
    # Multi-scale Fractal Dimension
    data['short_term_fractal'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['medium_term_fractal'] = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))
    data['fractal_ratio'] = data['short_term_fractal'] / data['medium_term_fractal']
    
    # Price Compression Detection
    data['range_compression'] = (data['fractal_ratio'] < 0.8).astype(int)
    data['range_expansion'] = (data['fractal_ratio'] > 1.2).astype(int)
    data['normal_range'] = ((data['fractal_ratio'] >= 0.8) & (data['fractal_ratio'] <= 1.2)).astype(int)
    
    # Volume Fractal Dynamics
    # Volume Pattern Recognition
    data['volume_change'] = data['volume'] - data['volume'].shift(1)
    data['volume_increase'] = (data['volume_change'] > 0).astype(int)
    
    # Volume Cluster Size
    data['volume_cluster_size'] = 0
    cluster_size = 0
    for i in range(len(data)):
        if i == 0:
            data.iloc[i, data.columns.get_loc('volume_cluster_size')] = 1
            cluster_size = 1
        else:
            if data['volume_increase'].iloc[i] == data['volume_increase'].iloc[i-1]:
                cluster_size += 1
            else:
                cluster_size = 1
            data.iloc[i, data.columns.get_loc('volume_cluster_size')] = cluster_size
    
    # Volume Cluster Intensity
    data['volume_cluster_intensity'] = 0
    for i in range(len(data)):
        if i >= data['volume_cluster_size'].iloc[i] - 1:
            start_idx = i - data['volume_cluster_size'].iloc[i] + 1
            cluster_sum = data['volume_change'].iloc[start_idx:i+1].sum()
            data.iloc[i, data.columns.get_loc('volume_cluster_intensity')] = cluster_sum
    
    # Volume Fractal Dimension
    data['volume_range_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_persistence'] = data['volume_cluster_size']
    data['volume_fractal_score'] = data['volume_range_ratio'] * data['volume_persistence']
    
    # Volume Regime Classification
    data['accumulation_regime'] = ((data['volume_cluster_size'] > 3) & 
                                  (data['volume_cluster_intensity'] > 0)).astype(int)
    data['distribution_regime'] = ((data['volume_cluster_size'] > 3) & 
                                  (data['volume_cluster_intensity'] < 0)).astype(int)
    data['neutral_regime'] = (data['volume_cluster_size'] <= 3).astype(int)
    
    # Momentum Fractal Structure
    # Multi-timeframe Momentum Fractals
    data['ultra_short_momentum'] = data['close'] / data['close'].shift(2) - 1
    data['short_term_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(13) - 1
    data['momentum_fractal_ratio'] = data['ultra_short_momentum'] / data['medium_term_momentum']
    
    # Momentum Regime Detection
    data['momentum_convergence'] = ((data['momentum_fractal_ratio'] >= 0.9) & 
                                   (data['momentum_fractal_ratio'] <= 1.1)).astype(int)
    data['momentum_divergence'] = ((data['momentum_fractal_ratio'] < 0.7) | 
                                  (data['momentum_fractal_ratio'] > 1.3)).astype(int)
    data['momentum_transition'] = (((data['momentum_fractal_ratio'] >= 0.7) & (data['momentum_fractal_ratio'] < 0.9)) | 
                                  ((data['momentum_fractal_ratio'] > 1.1) & (data['momentum_fractal_ratio'] <= 1.3))).astype(int)
    
    # Regime-Based Signal Generation
    # Compression Breakout Signals
    data['compression_breakout_signal'] = 0
    compression_mask = (data['range_compression'] == 1) & (data['accumulation_regime'] == 1)
    data.loc[compression_mask, 'compression_breakout_signal'] = (
        data.loc[compression_mask, 'volume_cluster_intensity'] * 
        data.loc[compression_mask, 'fractal_ratio']
    )
    
    # Expansion Continuation Signals
    data['expansion_continuation_signal'] = 0
    expansion_mask = (data['range_expansion'] == 1) & (data['distribution_regime'] == 0)
    data.loc[expansion_mask, 'expansion_continuation_signal'] = (
        data.loc[expansion_mask, 'momentum_fractal_ratio'] * 
        data.loc[expansion_mask, 'volume_fractal_score']
    )
    
    # Momentum Regime Integration
    # Convergence Breakout Setup
    data['convergence_breakout_signal'] = 0
    convergence_mask = (data['momentum_convergence'] == 1) & (data['accumulation_regime'] == 1)
    data.loc[convergence_mask, 'convergence_breakout_signal'] = (
        data.loc[convergence_mask, 'volume_cluster_size']
    )
    
    # Divergence Reversal Setup
    data['divergence_reversal_signal'] = 0
    divergence_mask = (data['momentum_divergence'] == 1) & (data['distribution_regime'] == 1)
    data.loc[divergence_mask, 'divergence_reversal_signal'] = (
        abs(data.loc[divergence_mask, 'momentum_fractal_ratio'] - 1)
    )
    
    # Fractal Quality Assessment
    # Volume-Price Alignment Score
    data['volume_price_alignment'] = data['volume_fractal_score'] * data['fractal_ratio']
    data['alignment_threshold'] = (data['volume_price_alignment'] > 1.0).astype(int)
    
    # Momentum-Volume Synchronization
    data['momentum_volume_sync'] = data['momentum_fractal_ratio'] * data['volume_fractal_score']
    data['synchronization_threshold'] = (data['momentum_volume_sync'] > 0.8).astype(int)
    data['desynchronization_warning'] = (data['momentum_volume_sync'] < 0.3).astype(int)
    
    # Regime Persistence Metrics
    # Current Regime Duration
    data['regime_duration'] = 0
    regime_dur = 0
    current_regime = 0
    for i in range(len(data)):
        if i == 0:
            regime_dur = 1
            current_regime = 0  # neutral
        else:
            current_regime_type = 0
            if data['accumulation_regime'].iloc[i] == 1:
                current_regime_type = 1
            elif data['distribution_regime'].iloc[i] == 1:
                current_regime_type = 2
            
            prev_regime_type = 0
            if data['accumulation_regime'].iloc[i-1] == 1:
                prev_regime_type = 1
            elif data['distribution_regime'].iloc[i-1] == 1:
                prev_regime_type = 2
            
            if current_regime_type == prev_regime_type:
                regime_dur += 1
            else:
                regime_dur = 1
            
        data.iloc[i, data.columns.get_loc('regime_duration')] = regime_dur
    
    data['regime_stability'] = data['regime_duration'] * data['volume_cluster_intensity']
    
    # Fractal Consistency Score
    data['price_fractal_consistency'] = data['fractal_ratio'].rolling(window=5, min_periods=1).std()
    data['volume_fractal_consistency'] = data['volume_fractal_score'].rolling(window=5, min_periods=1).std()
    data['overall_consistency'] = data['price_fractal_consistency'] * data['volume_fractal_consistency']
    
    # Final Alpha Factor Calculation
    # Combine all signals with quality assessment
    data['final_alpha'] = (
        data['compression_breakout_signal'] * data['alignment_threshold'] +
        data['expansion_continuation_signal'] * data['synchronization_threshold'] +
        data['convergence_breakout_signal'] * (1 - data['desynchronization_warning']) +
        data['divergence_reversal_signal'] * data['regime_stability']
    ) / (1 + data['overall_consistency'])
    
    # Handle infinite values and NaN
    data['final_alpha'] = data['final_alpha'].replace([np.inf, -np.inf], np.nan)
    data['final_alpha'] = data['final_alpha'].fillna(0)
    
    return data['final_alpha']
