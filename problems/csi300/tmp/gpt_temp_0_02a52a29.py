import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Volume-Weighted Momentum
    # Volume-Adjusted Momentum Divergence
    data['ultra_short_momentum'] = (data['close'] - data['close'].shift(2)) * data['volume']
    data['short_term_momentum'] = (data['close'] - data['close'].shift(5)) * data['volume'].rolling(window=3, min_periods=1).mean()
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(10)) * data['volume'].rolling(window=6, min_periods=1).mean()
    data['volume_momentum_divergence'] = (data['ultra_short_momentum'] - data['short_term_momentum']) / (1 + np.abs(data['medium_term_momentum']))
    
    # Volume-Price Efficiency Analysis
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_efficiency'] = data['volume'] / data['amount'].replace(0, np.nan)
    data['volume_price_alignment'] = data['price_efficiency'] * data['volume_efficiency']
    data['efficiency_persistence'] = data['volume_price_alignment'].rolling(window=3, min_periods=1).sum()
    
    # Momentum-Volume Regime Detection
    data['volume_regime_indicator'] = data['volume'] / data['volume'].shift(1).rolling(window=10, min_periods=1).median()
    data['momentum_regime_indicator'] = np.sign(data['volume_momentum_divergence']) * np.sign(data['volume_regime_indicator'] - 1)
    data['regime_persistence'] = (data['momentum_regime_indicator'].rolling(window=5, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] and not np.isnan(x.iloc[i]) and not np.isnan(x.iloc[i-1])])
    ))
    
    # Gap Absorption with Volume Confirmation
    # Gap Analysis with Volume Context
    data['opening_gap'] = data['open'] - data['close'].shift(1)
    data['gap_volume_intensity'] = data['opening_gap'] * data['volume']
    data['gap_absorption_ratio'] = (data['close'] - data['open']) / data['opening_gap'].replace(0, np.nan)
    data['volume_confirmed_gap'] = data['gap_absorption_ratio'] * data['volume']
    
    # Intraday Volume-Pressure Dynamics
    data['high_side_volume_pressure'] = (data['high'] - data['open']) * data['volume']
    data['low_side_volume_pressure'] = (data['open'] - data['low']) * data['volume']
    data['volume_pressure_imbalance'] = data['high_side_volume_pressure'] / data['low_side_volume_pressure'].replace(0, np.nan)
    data['pressure_volume_alignment'] = data['volume_pressure_imbalance'] * data['volume_efficiency']
    
    # Gap-Volume Convergence
    data['gap_volume_momentum'] = data['volume_confirmed_gap'] * data['volume_momentum_divergence']
    data['pressure_gap_coherence'] = np.sign(data['gap_volume_momentum']) * np.sign(data['pressure_volume_alignment'])
    data['gap_absorption_persistence'] = data['gap_volume_momentum'].rolling(window=3, min_periods=1).mean()
    
    # Volume-Clustered Reversal Patterns
    # Volume Spike Reversal Detection
    data['volume_spike_threshold'] = (data['volume'] > 2 * data['volume'].shift(1).rolling(window=15, min_periods=1).median()).astype(int)
    data['price_reversal_signal'] = (np.sign(data['close'] - data['close'].shift(1)) != np.sign(data['close'].shift(1) - data['close'].shift(2))).astype(int)
    data['volume_spike_reversal'] = data['volume_spike_threshold'] * data['price_reversal_signal'] * data['opening_gap']
    
    # Volume-Weighted Exhaustion Signals
    data['high_volume_rejection'] = ((data['high'] == data['open']) & (data['close'] < data['open'])).astype(int) * data['volume']
    data['low_volume_absorption'] = ((data['low'] == data['open']) & (data['close'] > data['open'])).astype(int) * data['volume']
    data['volume_exhaustion_divergence'] = data['high_volume_rejection'] - data['low_volume_absorption']
    
    # Volume-Cluster Timing
    data['volume_cluster_detection'] = ((data['volume'] > data['volume'].shift(1).rolling(window=5, min_periods=1).mean()) & 
                                      (data['volume'].shift(1) > data['volume'].shift(2).rolling(window=5, min_periods=1).mean())).astype(int)
    data['cluster_reversal_timing'] = data['volume_cluster_detection'] * data['volume_spike_reversal']
    data['volume_cluster_momentum'] = data['cluster_reversal_timing'] * data['volume_momentum_divergence']
    
    # Regime-Dependent Volume Efficiency
    # Volume-Flow Efficiency Metrics
    data['amount_per_volume_efficiency'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['price_range_volume_efficiency'] = (data['high'] - data['low']) / data['volume'].replace(0, np.nan)
    data['volume_flow_alignment'] = data['amount_per_volume_efficiency'] * data['price_range_volume_efficiency']
    
    # Volume Regime Classification
    data['high_volume_regime'] = (data['volume'] > 1.5 * data['volume'].shift(1).rolling(window=20, min_periods=1).median()).astype(int)
    data['low_volume_regime'] = (data['volume'] < 0.7 * data['volume'].shift(1).rolling(window=20, min_periods=1).median()).astype(int)
    data['regime_efficiency_multiplier'] = data['high_volume_regime'] - data['low_volume_regime']
    
    # Regime-Adaptive Volume Factors
    data['regime_volume_efficiency'] = data['volume_flow_alignment'] * data['regime_efficiency_multiplier']
    data['volume_regime_persistence'] = (data['high_volume_regime'].rolling(window=5, min_periods=1).sum() + 
                                       data['low_volume_regime'].rolling(window=5, min_periods=1).sum())
    data['adaptive_volume_momentum'] = data['volume_momentum_divergence'] * data['regime_volume_efficiency']
    
    # Composite Volume-Asymmetry Alpha
    # Core Volume-Momentum Factor
    data['base_volume_momentum'] = data['volume_momentum_divergence'] * data['volume_price_alignment']
    data['gap_enhanced_momentum'] = data['base_volume_momentum'] * (1 + 0.3 * data['gap_volume_momentum'])
    data['regime_adapted_volume'] = data['gap_enhanced_momentum'] * data['adaptive_volume_momentum']
    
    # Volume-Pressure Timing Layer
    data['pressure_volume_adjustment'] = data['regime_adapted_volume'] * data['pressure_volume_alignment']
    data['cluster_timing_validation'] = data['pressure_volume_adjustment'] * (1 + 0.2 * data['volume_cluster_momentum'])
    data['exhaustion_confirmation'] = data['cluster_timing_validation'] * (1 + 0.15 * data['volume_exhaustion_divergence'])
    
    # Efficiency-Weighted Application
    data['volume_efficiency_factor'] = data['exhaustion_confirmation'] * data['volume_flow_alignment']
    data['regime_persistence_weighted'] = data['volume_efficiency_factor'] * data['volume_regime_persistence']
    data['efficiency_persistent_alpha'] = data['regime_persistence_weighted'] * data['efficiency_persistence']
    
    # Final Volume-Asymmetry Signal
    alpha_signal = (data['volume_momentum_divergence'] * 
                   data['volume_confirmed_gap'] * 
                   data['volume_spike_reversal'])
    
    # Combine all components with appropriate weights
    final_alpha = (0.4 * alpha_signal + 
                  0.3 * data['efficiency_persistent_alpha'] + 
                  0.2 * data['gap_absorption_persistence'] + 
                  0.1 * data['regime_persistence'])
    
    return final_alpha
