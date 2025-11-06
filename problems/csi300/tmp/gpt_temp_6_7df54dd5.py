import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Helper function for True Range
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = true_range(data['high'], data['low'], data['prev_close'])
    data['range_entanglement'] = (data['high'] - data['low']) / data['true_range']
    data['efficiency_coherence'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume quantum states
    data['volume_quantum_states'] = (data['volume'] / data['volume'].shift(1) - 1).fillna(0)
    
    # Momentum persistence
    data['momentum_persistence'] = np.nan
    for i in range(3, len(data)):
        window = data.iloc[i-3:i+1]
        if len(window) == 4:
            min_low = window['low'].min()
            max_high = window['high'].max()
            if max_high > min_low:
                data.iloc[i, data.columns.get_loc('momentum_persistence')] = (data.iloc[i]['close'] - min_low) / (max_high - min_low)
    
    # Volume momentum singularities
    data['volume_median_3d'] = data['volume'].rolling(window=3, min_periods=1).median()
    data['volume_singularity'] = data['volume'] / data['volume_median_3d']
    
    # Volatility ratio
    data['volatility_ratio'] = data['true_range'] / data['true_range'].rolling(window=5, min_periods=1).mean()
    
    # Momentum components
    data['short_momentum'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['medium_momentum'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    data['momentum_convergence'] = data['short_momentum'] * data['medium_momentum']
    
    # Intraday patterns
    data['morning_strength'] = (data['high'] - data['open']) / data['open']
    data['afternoon_support'] = (data['close'] - data['low']) / data['low']
    data['morning_efficiency'] = (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['afternoon_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Quantum Range Efficiency Component
    data['quantum_range_stability'] = data['range_entanglement'] * data['efficiency_coherence']
    data['liquidity_quantum_states'] = data['volume_quantum_states'] * data['volume_singularity']
    data['quantum_range_liquidity'] = data['quantum_range_stability'] * data['liquidity_quantum_states']
    
    # Topological Momentum Component
    data['adaptive_topological_momentum'] = data['momentum_persistence'] * data['volume_singularity']
    data['volume_validated_topology'] = data['adaptive_topological_momentum'] * data['momentum_persistence']
    data['topological_momentum_strength'] = data['volume_validated_topology'] * data['momentum_convergence']
    
    # Volatility-Momentum Quantum Component
    data['volatility_weighted_momentum'] = data['momentum_convergence'] * data['volatility_ratio']
    data['quantum_volatility_momentum'] = data['volatility_weighted_momentum'] * data['momentum_convergence']
    data['volatility_momentum_entanglement'] = data['quantum_volatility_momentum'] * data['momentum_convergence']
    data['quantum_transition_strength'] = data['volatility_momentum_entanglement'] * data['volatility_ratio']
    
    # Intraday Quantum-Topology
    data['intraday_quantum_efficiency'] = data['morning_strength'] * data['afternoon_support']
    data['intraday_topological_alignment'] = data['intraday_quantum_efficiency'] * data['morning_efficiency']
    
    # Core Quantum-Topological Integration
    data['quantum_range_momentum'] = data['quantum_range_liquidity'] * data['topological_momentum_strength']
    data['volatility_quantum_topology'] = data['quantum_transition_strength'] * data['intraday_topological_alignment']
    data['quantum_topological_integration'] = data['quantum_range_momentum'] * data['volatility_quantum_topology']
    
    # Multi-scale validation components
    data['quantum_scale_coherence'] = data['range_entanglement'].rolling(window=5).std() * data['momentum_persistence'].rolling(window=5).mean()
    data['quantum_topological_convergence'] = data['quantum_scale_coherence'] * data['topological_momentum_strength']
    
    # Final alpha construction
    data['quantum_scale_validation'] = data['quantum_topological_integration'] * data['quantum_topological_convergence']
    data['quantum_topological_stability'] = data['quantum_scale_validation'] * data['range_entanglement']
    
    data['base_quantum_factor'] = data['quantum_topological_integration'] * data['quantum_scale_validation']
    data['quantum_topological_enhancement'] = data['base_quantum_factor'] * data['quantum_topological_stability']
    data['final_alpha'] = data['quantum_topological_enhancement'] * data['intraday_quantum_efficiency']
    
    # Return the final alpha factor
    return data['final_alpha']
