import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Price Structure Analysis
    # Multi-Scale Price Fractal Dimension
    data['price_diff'] = data['close'].diff()
    data['abs_price_diff'] = data['price_diff'].abs()
    
    # Short-Term Fractal
    data['short_term_fractal'] = np.log(data['abs_price_diff'].rolling(window=5).sum()) / np.log(5)
    
    # Medium-Term Fractal
    data['medium_term_fractal'] = np.log(data['abs_price_diff'].rolling(window=20).sum()) / np.log(20)
    
    # Fractal Ratio
    data['fractal_ratio'] = data['short_term_fractal'] / data['medium_term_fractal']
    
    # Fractal Momentum
    data['fractal_momentum'] = data['short_term_fractal'] - data['short_term_fractal'].shift(5)
    
    # Price Path Efficiency
    data['actual_path_length'] = data['abs_price_diff'].rolling(window=5).sum()
    data['straight_line_distance'] = (data['close'] - data['close'].shift(5)).abs()
    data['path_efficiency'] = data['straight_line_distance'] / data['actual_path_length']
    data['efficiency_momentum'] = data['path_efficiency'] - data['path_efficiency'].shift(5)
    
    # Fractal-Efficiency Divergence
    data['fractal_expansion'] = data['fractal_ratio'] * data['path_efficiency']
    data['efficiency_fractal_gap'] = data['path_efficiency'] - data['fractal_ratio']
    data['fractal_persistence'] = data['fractal_ratio'].rolling(window=10).apply(lambda x: (x > 1).sum()) / 10
    
    # Volume Fractality & Microstructure Entropy
    # Volume Fractal Structure
    data['volume_diff'] = data['volume'].diff()
    data['abs_volume_diff'] = data['volume_diff'].abs()
    
    data['volume_fractal_dimension'] = np.log(data['abs_volume_diff'].rolling(window=5).sum()) / np.log(5)
    data['volume_path_efficiency'] = (data['volume'] - data['volume'].shift(5)).abs() / data['abs_volume_diff'].rolling(window=5).sum()
    data['volume_price_fractal_alignment'] = data['volume_fractal_dimension'] / data['short_term_fractal']
    data['volume_fractal_momentum'] = data['volume_fractal_dimension'] - data['volume_fractal_dimension'].shift(5)
    
    # Microstructure Entropy Analysis
    def calculate_entropy(series, window):
        def entropy_calc(x):
            x_sum = x.sum()
            if x_sum == 0:
                return 0
            probs = x / x_sum
            probs = probs[probs > 0]  # Remove zeros to avoid log(0)
            return -np.sum(probs * np.log(probs))
        return series.rolling(window=window).apply(entropy_calc, raw=True)
    
    data['price_volume_entropy'] = calculate_entropy(data['volume'], 5)
    data['range'] = data['high'] - data['low']
    data['range_entropy'] = calculate_entropy(data['range'], 5)
    data['entropy_ratio'] = data['price_volume_entropy'] / data['range_entropy']
    data['entropy_momentum'] = data['price_volume_entropy'] - data['price_volume_entropy'].shift(5)
    
    # Fractal-Entropy Synchronization
    data['fractal_entropy_alignment'] = data['fractal_ratio'] * data['entropy_ratio']
    data['volume_entropy_divergence'] = data['volume_fractal_dimension'] - data['price_volume_entropy']
    data['entropy_persistence'] = data['entropy_ratio'].rolling(window=5).apply(lambda x: (x > 1).sum()) / 5
    
    # Regime-Based Fractal Classification
    data['high_fractal_regime'] = (data['fractal_ratio'] > 1.2) & (data['volume_fractal_dimension'] > 1.5)
    data['low_fractal_regime'] = (data['fractal_ratio'] < 0.8) & (data['volume_fractal_dimension'] < 1.2)
    data['transitional_regime'] = (data['fractal_ratio'] >= 0.8) & (data['fractal_ratio'] <= 1.2)
    
    data['high_entropy_regime'] = (data['entropy_ratio'] > 1.1) & (data['price_volume_entropy'] > 1.5)
    data['low_entropy_regime'] = (data['entropy_ratio'] < 0.9) & (data['price_volume_entropy'] < 1.2)
    data['balanced_entropy_regime'] = (data['entropy_ratio'] >= 0.9) & (data['entropy_ratio'] <= 1.1)
    
    # Fractal-Entropy Regime Matrix
    data['chaotic_regime'] = data['high_fractal_regime'] & data['high_entropy_regime']
    data['structured_regime'] = data['low_fractal_regime'] & data['low_entropy_regime']
    data['fractal_driven_regime'] = data['high_fractal_regime'] & data['low_entropy_regime']
    data['entropy_driven_regime'] = data['low_fractal_regime'] & data['high_entropy_regime']
    
    # Intraday Microstructure Fractality
    # Opening Fractal Analysis
    data['opening_gap_fractal'] = (data['open'] - data['close'].shift(1)).abs() / data['range']
    data['opening_efficiency'] = (data['high'] - data['open']) / data['range']
    data['opening_pressure'] = (data['open'] - data['low']) / data['range']
    data['opening_fractal_momentum'] = data['opening_gap_fractal'] - data['opening_gap_fractal'].shift(3)
    
    # Closing Microstructure
    data['closing_range_position'] = (data['close'] - data['low']) / data['range']
    data['closing_efficiency'] = (data['close'] - data['open']).abs() / data['range']
    data['closing_fractal'] = np.log((data['close'] - data['close'].shift(1)).abs()) / np.log(data['range'])
    data['closing_pressure_momentum'] = data['closing_range_position'] - data['closing_range_position'].shift(3)
    
    # Intraday Fractal Integration
    data['open_close_fractal_alignment'] = data['opening_gap_fractal'] * data['closing_fractal']
    data['intraday_efficiency_divergence'] = data['opening_efficiency'] - data['closing_efficiency']
    data['microstructure_fractal_persistence'] = (data['closing_fractal'] > data['opening_gap_fractal']).rolling(window=5).sum() / 5
    
    # Multi-Scale Fractal Momentum
    # Fractal Momentum Components
    data['price_fractal_momentum'] = data['fractal_momentum'] * data['path_efficiency']
    data['volume_fractal_momentum'] = data['volume_fractal_momentum'] * data['volume_path_efficiency']
    data['entropy_momentum_component'] = data['entropy_momentum'] * data['entropy_ratio']
    data['combined_fractal_momentum'] = data['price_fractal_momentum'] + data['volume_fractal_momentum'] + data['entropy_momentum_component']
    
    # Fractal Reversal Detection
    data['fractal_expansion_reversal'] = data['fractal_expansion'] * data['efficiency_fractal_gap']
    data['volume_price_fractal_divergence'] = data['volume_fractal_momentum'] - data['price_fractal_momentum']
    data['entropy_fractal_reversal'] = data['entropy_momentum'] * data['fractal_entropy_alignment']
    data['multi_fractal_reversal_strength'] = data['volume_price_fractal_divergence'].abs() * data['entropy_fractal_reversal'].abs()
    
    # Fractal Momentum Integration
    data['core_fractal_signal'] = data['combined_fractal_momentum'] * data['fractal_persistence']
    data['fractal_reversal_component'] = data['fractal_expansion_reversal'] * data['multi_fractal_reversal_strength']
    data['fractal_momentum_reversal_ratio'] = data['core_fractal_signal'] / (1 + data['fractal_reversal_component'].abs())
    
    # Fractal-Entropy Synchronization Signals
    data['strong_fractal_alignment'] = (data['fractal_entropy_alignment'] > 0) & (data['volume_price_fractal_alignment'] > 1)
    data['weak_fractal_alignment'] = (data['fractal_entropy_alignment'] < 0) | (data['volume_price_fractal_alignment'] < 1)
    data['fractal_synchronization_strength'] = data['fractal_entropy_alignment'].abs() * data['volume_price_fractal_alignment']
    
    data['high_microstructure_fractal'] = (data['opening_gap_fractal'] > 0.7) & (data['closing_fractal'] > 0.6)
    data['low_microstructure_fractal'] = (data['opening_gap_fractal'] < 0.3) & (data['closing_fractal'] < 0.4)
    data['intraday_fractal_consistency'] = (data['intraday_efficiency_divergence'] > 0).rolling(window=5).sum() / 5
    
    # Adaptive Signal Construction
    # Core Fractal Alpha
    data['core_fractal_alpha'] = data['combined_fractal_momentum'] * data['fractal_synchronization_strength']
    data['microstructure_enhancement'] = data['open_close_fractal_alignment'] * data['intraday_fractal_consistency']
    data['entropy_multiplier'] = data['entropy_ratio'] * data['entropy_persistence']
    
    # Base Fractal Alpha Construction
    data['core_signal'] = data['core_fractal_alpha'] * data['microstructure_enhancement']
    data['entropy_adjustment'] = data['core_signal'] * data['entropy_multiplier']
    data['base_alpha'] = data['entropy_adjustment'] * data['fractal_momentum_reversal_ratio']
    
    # Regime-Adaptive Refinement
    data['chaotic_refinement'] = data['base_alpha'] * (2 - data['fractal_ratio']) * data['entropy_ratio']
    data['structured_refinement'] = data['base_alpha'] * data['fractal_ratio'] * data['path_efficiency']
    data['fractal_driven_refinement'] = data['base_alpha'] * data['fractal_expansion'] * data['opening_fractal_momentum']
    data['entropy_driven_refinement'] = data['base_alpha'] * data['entropy_momentum'] * data['closing_pressure_momentum']
    
    # Final Alpha Output with Regime-Specific Refinement
    data['final_alpha'] = np.where(data['chaotic_regime'], data['chaotic_refinement'],
                          np.where(data['structured_regime'], data['structured_refinement'],
                          np.where(data['fractal_driven_regime'], data['fractal_driven_refinement'],
                          np.where(data['entropy_driven_regime'], data['entropy_driven_refinement'],
                          data['base_alpha']))))
    
    # Incorporate Fractal Persistence and Cross-Fractal Validation
    data['final_alpha'] = data['final_alpha'] * data['fractal_persistence']
    data['final_alpha'] = data['final_alpha'] * data['intraday_fractal_consistency'] * data['volume_fractal_momentum']
    
    # Return the final alpha factor
    return data['final_alpha']
