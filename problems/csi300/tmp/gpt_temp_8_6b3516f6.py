import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Identification
    # Multi-Scale Volatility Assessment
    data['short_term_vol'] = data['close'].rolling(window=5).std() / data['close'].rolling(window=5).mean()
    data['medium_term_vol'] = data['close'].rolling(window=20).std() / data['close'].rolling(window=20).mean()
    data['vol_regime_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    
    # Regime Classification
    data['high_vol_regime'] = (data['vol_regime_ratio'] > 1.2).astype(int)
    data['low_vol_regime'] = (data['vol_regime_ratio'] < 0.8).astype(int)
    
    # Regime Stability
    regime_cols = ['high_vol_regime', 'low_vol_regime']
    for col in regime_cols:
        data[f'{col}_stability'] = data[col].rolling(window=5).sum() / 5
    
    # Volatility Momentum
    data['vol_acceleration'] = (data['short_term_vol'] - data['medium_term_vol']) / (data['short_term_vol'] + 1e-8)
    data['regime_momentum'] = data['vol_acceleration'] * data['high_vol_regime_stability']
    
    # Price-Volume Entropy Analysis
    # Information Efficiency Metrics
    def calculate_entropy(series, window=5):
        entropy_values = []
        for i in range(len(series)):
            if i < window - 1:
                entropy_values.append(np.nan)
                continue
            window_data = series.iloc[i-window+1:i+1]
            if window_data.std() == 0:
                entropy_values.append(0)
                continue
            # Normalize and create probability distribution
            normalized = (window_data - window_data.min()) / (window_data.max() - window_data.min() + 1e-8)
            probabilities = normalized / (normalized.sum() + 1e-8)
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
            entropy_values.append(entropy)
        return pd.Series(entropy_values, index=series.index)
    
    data['price_entropy'] = calculate_entropy(data['close'])
    data['volume_entropy'] = calculate_entropy(data['volume'])
    data['joint_entropy'] = data['price_entropy'] * data['volume_entropy']
    
    # Entropy Divergence Patterns
    data['price_volume_divergence'] = abs(data['price_entropy'] - data['volume_entropy']) / (data['price_entropy'] + data['volume_entropy'] + 1e-8)
    data['entropy_momentum'] = (data['joint_entropy'] - data['joint_entropy'].shift(4)) / (data['joint_entropy'].shift(4) + 1e-8)
    data['information_efficiency'] = 1 / (data['price_volume_divergence'] + 0.001)
    
    # Microstructure Entropy
    data['gap_entropy'] = abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    data['intraday_entropy'] = (data['high'] - data['low']) / (data['open'] + data['close'] + 1e-8)
    data['total_micro_entropy'] = data['gap_entropy'] * data['intraday_entropy']
    
    # Fractal Momentum Structure
    # Multi-Fractal Price Patterns
    data['short_fractal'] = data['close'] / data['close'].shift(2) - 1
    data['medium_fractal'] = data['close'] / data['close'].shift(8) - 1
    data['long_fractal'] = data['close'] / data['close'].shift(21) - 1
    data['fractal_ratio'] = (data['short_fractal'] + data['medium_fractal']) / (data['long_fractal'] + 0.001)
    
    # Fractal Momentum Convergence
    data['fractal_alignment'] = np.sign(data['short_fractal']) * np.sign(data['medium_fractal']) * np.sign(data['long_fractal'])
    data['fractal_momentum_quality'] = data['fractal_ratio'] * data['fractal_alignment']
    
    # Fractal Stability
    data['fractal_stability'] = data['fractal_alignment'].rolling(window=5).apply(
        lambda x: (x == x.iloc[0]).sum() / 5 if not x.isna().any() else np.nan, raw=False
    )
    
    # Volatility-Adjusted Fractals
    data['high_vol_fractal'] = data['fractal_momentum_quality'] * data['short_term_vol']
    data['low_vol_fractal'] = data['fractal_momentum_quality'] / (data['short_term_vol'] + 0.001)
    data['regime_adaptive_fractal'] = data['high_vol_fractal'] * data['high_vol_regime_stability']
    
    # Entropy-Momentum Integration
    # Information-Momentum Alignment
    data['entropy_momentum_correlation'] = np.sign(data['fractal_momentum_quality']) * np.sign(data['entropy_momentum'])
    data['alignment_strength'] = data['fractal_momentum_quality'] * data['entropy_momentum_correlation']
    data['information_efficiency_score'] = data['alignment_strength'] * data['information_efficiency']
    
    # Regime-Adaptive Weighting
    data['high_vol_combination'] = 0.7 * data['information_efficiency_score'] + 0.3 * data['regime_adaptive_fractal']
    data['low_vol_combination'] = 0.3 * data['information_efficiency_score'] + 0.7 * data['regime_adaptive_fractal']
    data['transition_factor'] = data['high_vol_regime_stability'] * data['alignment_strength']
    
    # Microstructure Enhancement
    data['base_integration'] = data['high_vol_combination'] * data['transition_factor']
    data['entropy_adjustment'] = data['base_integration'] * data['total_micro_entropy']
    data['avg_volume_ratio'] = data['volume'].rolling(window=5).mean() / (data['volume'] + 1e-8)
    data['volume_context'] = data['entropy_adjustment'] / data['avg_volume_ratio']
    
    # Composite Alpha Generation
    data['core_alpha'] = data['volume_context'] * data['information_efficiency_score']
    data['fractal_scaling'] = data['core_alpha'] * data['fractal_stability']
    data['final_alpha'] = data['fractal_scaling'] * data['regime_momentum']
    
    return data['final_alpha']
