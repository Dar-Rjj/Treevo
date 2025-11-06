import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Directional Volatility Asymmetry Analysis
    # Upward Volatility Component
    data['up_range'] = data['high'] - data['close']
    data['total_range'] = data['high'] - data['low']
    data['up_intensity'] = data['up_range'] / data['total_range']
    
    # Downward Volatility Component
    data['down_range'] = data['close'] - data['low']
    data['down_intensity'] = data['down_range'] / data['total_range']
    
    # Volatility Persistence (5-day rolling)
    data['up_persistence'] = data['up_range'].rolling(window=5).sum() / data['total_range'].rolling(window=5).sum()
    data['down_persistence'] = data['down_range'].rolling(window=5).sum() / data['total_range'].rolling(window=5).sum()
    
    # Volatility Asymmetry Ratio
    data['vol_asymmetry_ratio'] = (data['up_persistence'] / data['down_persistence']) - 1
    
    # Volatility Skew Momentum
    data['asymmetry_momentum'] = (data['vol_asymmetry_ratio'] / data['vol_asymmetry_ratio'].shift(5)) - 1
    
    # Asymmetric Volatility Signal
    data['asym_vol_signal'] = data['vol_asymmetry_ratio'] * data['asymmetry_momentum']
    
    # Bidirectional Flow Momentum System
    # Opening Flow Strength
    data['open_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['open_vol_intensity'] = data['volume'] / data['volume'].shift(1)
    data['open_flow_power'] = data['open_momentum'] * data['open_vol_intensity']
    
    # Closing Flow Strength
    data['close_momentum'] = (data['close'] - data['open']) / data['open']
    data['close_vol_intensity'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['close_flow_power'] = data['close_momentum'] * data['close_vol_intensity']
    
    # Flow Reversal Patterns
    data['flow_divergence'] = abs(data['open_flow_power'] - data['close_flow_power'])
    data['flow_reversal_prob'] = data['flow_divergence'] / (abs(data['open_flow_power']) + abs(data['close_flow_power']))
    
    # Flow Persistence (3-day consistency)
    def flow_direction(x):
        return 1 if x > 0 else (-1 if x < 0 else 0)
    
    data['open_flow_dir'] = data['open_flow_power'].apply(flow_direction)
    data['close_flow_dir'] = data['close_flow_power'].apply(flow_direction)
    
    data['flow_consistency'] = (data['open_flow_dir'].rolling(window=3).apply(lambda x: (x == x.iloc[0]).sum() / len(x), raw=False) +
                               data['close_flow_dir'].rolling(window=3).apply(lambda x: (x == x.iloc[0]).sum() / len(x), raw=False)) / 2
    
    data['flow_momentum_decay'] = (abs(data['open_flow_power']) + abs(data['close_flow_power'])) * data['flow_consistency']
    
    # Bidirectional Flow Signal
    data['bidirectional_flow_signal'] = data['flow_reversal_prob'] * data['flow_momentum_decay']
    
    # Volume-Price Fractal Alignment
    # Short-term Volume Fractal
    data['vol_range_3d'] = data['volume'].rolling(window=3).max() - data['volume'].rolling(window=3).min()
    data['vol_path_length'] = abs(data['volume'] - data['volume'].shift(1)).rolling(window=2).sum()
    data['vol_fractal'] = np.log(data['vol_range_3d']) / np.log(data['vol_path_length'])
    
    # Medium-term Price-Volume Fractal
    data['combined_range'] = (abs(data['high'] - data['low']) * data['volume']).rolling(window=8).sum()
    data['combined_path'] = (abs(data['close'] - data['close'].shift(1)) * abs(data['volume'] - data['volume'].shift(1))).rolling(window=8).sum()
    data['price_vol_fractal'] = np.log(data['combined_range']) / np.log(data['combined_path'])
    
    # Fractal Alignment Divergence
    data['fractal_correlation'] = data['vol_fractal'] * data['price_vol_fractal']
    data['fractal_divergence'] = data['vol_fractal'] - data['price_vol_fractal']
    data['divergence_momentum'] = (data['fractal_divergence'] / data['fractal_divergence'].shift(5)) - 1
    
    # Fractal Alignment Signal
    data['fractal_alignment_signal'] = data['fractal_correlation'] * data['divergence_momentum']
    
    # Session Boundary Transition Analysis
    # Overnight Momentum Carry
    data['overnight_return'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['overnight_vol_ratio'] = data['volume'] / data['volume'].shift(1)
    data['overnight_momentum'] = data['overnight_return'] * data['overnight_vol_ratio']
    
    # Intraday Transition Efficiency
    data['session_transition'] = abs(data['close'] - data['open']) / abs(data['high'] - data['low'])
    data['transition_consistency'] = data['session_transition'] / data['session_transition'].rolling(window=5).mean()
    data['transition_efficiency'] = data['session_transition'] * data['transition_consistency']
    
    # Boundary Reversal Patterns
    data['boundary_divergence'] = abs(data['overnight_momentum'] - data['close_momentum'])
    data['boundary_reversal_prob'] = data['boundary_divergence'] / (abs(data['overnight_momentum']) + abs(data['close_momentum']))
    
    # Boundary Persistence (5-day consistency)
    def boundary_direction(x):
        return 1 if x > 0 else (-1 if x < 0 else 0)
    
    data['overnight_dir'] = data['overnight_momentum'].apply(boundary_direction)
    data['intraday_dir'] = data['close_momentum'].apply(boundary_direction)
    
    data['boundary_consistency'] = (data['overnight_dir'].rolling(window=5).apply(lambda x: (x == x.iloc[0]).sum() / len(x), raw=False) +
                                   data['intraday_dir'].rolling(window=5).apply(lambda x: (x == x.iloc[0]).sum() / len(x), raw=False)) / 2
    
    data['boundary_momentum_transfer'] = data['overnight_momentum'] * data['transition_efficiency'] * data['boundary_consistency']
    
    # Session Transition Signal
    data['session_transition_signal'] = data['boundary_reversal_prob'] * data['boundary_momentum_transfer']
    
    # Synthesize Final Alpha Signal
    # Base Asymmetry Signal
    data['base_asymmetry'] = data['asym_vol_signal'] * data['bidirectional_flow_signal'] * data['fractal_alignment_signal']
    
    # Incorporate Session Transition Dynamics
    data['enhanced_signal'] = data['base_asymmetry'] * data['session_transition_signal'] * data['boundary_consistency']
    
    # Flow-Volatility Integration
    data['flow_vol_correlation'] = data['bidirectional_flow_signal'].rolling(window=5).corr(data['asym_vol_signal'])
    data['final_alpha'] = data['enhanced_signal'] * data['flow_vol_correlation']
    
    # Return the final alpha factor
    return data['final_alpha']
