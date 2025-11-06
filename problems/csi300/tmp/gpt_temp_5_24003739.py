import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Small epsilon to avoid division by zero
    eps = 1e-8
    
    # 1. Asymmetric Volatility Framework
    # Directional Volatility Components
    data['upward_vol'] = (data['high'] - data['open']) / (data['high'] - data['low'] + eps)
    data['downward_vol'] = (data['open'] - data['low']) / (data['high'] - data['low'] + eps)
    data['vol_asymmetry'] = data['upward_vol'] - data['downward_vol']
    
    # Multi-Timeframe Volatility Patterns
    data['short_term_asymmetry'] = data['vol_asymmetry'] - data['vol_asymmetry'].shift(1)
    data['medium_term_asymmetry'] = data['vol_asymmetry'].rolling(window=3, min_periods=1).mean() - \
                                   data['vol_asymmetry'].shift(3).rolling(window=3, min_periods=1).mean()
    data['asymmetry_momentum'] = data['short_term_asymmetry'] * data['medium_term_asymmetry']
    
    # 2. Volume-Price Distribution Dynamics
    # Volume Concentration Analysis
    volume_avg_5 = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_efficiency'] = data['volume'] / (volume_avg_5 + eps)
    data['price_impact_efficiency'] = (data['close'] - data['open']) / (data['volume'] + eps)
    data['concentration_divergence'] = data['volume_efficiency'] - data['price_impact_efficiency']
    
    # Distribution Shape Detection
    volume_std_5 = data['volume'].rolling(window=5, min_periods=1).std()
    data['volume_skewness'] = (data['volume'] - volume_avg_5) / (volume_std_5 + eps)
    data['price_distribution'] = (data['high'] - data['close']) / (data['close'] - data['low'] + eps)
    data['distribution_alignment'] = data['volume_skewness'] * data['price_distribution']
    
    # 3. Momentum Fracture Detection
    # Price Momentum Discontinuities
    data['momentum_gap'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(2) + eps)
    data['acceleration_fracture'] = data['momentum_gap'] - data['momentum_gap'].shift(1)
    
    # Momentum Persistence
    price_diff_1 = data['close'] - data['close'].shift(1)
    price_diff_2 = data['close'].shift(1) - data['close'].shift(2)
    price_diff_3 = data['close'].shift(2) - data['close'].shift(3)
    
    momentum_persistence = []
    for i in range(len(data)):
        if i < 3:
            momentum_persistence.append(0)
        else:
            count = 0
            if np.sign(price_diff_1.iloc[i]) == np.sign(price_diff_2.iloc[i]):
                count += 1
            if np.sign(price_diff_2.iloc[i]) == np.sign(price_diff_3.iloc[i]):
                count += 1
            if np.sign(price_diff_1.iloc[i]) == np.sign(price_diff_3.iloc[i]):
                count += 1
            momentum_persistence.append(count / 3)
    data['momentum_persistence'] = momentum_persistence
    
    # Volume Momentum Patterns
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_acceleration'] = data['volume_momentum'] - data['volume_momentum'].shift(1)
    data['volume_price_momentum_divergence'] = data['volume_acceleration'] - data['acceleration_fracture']
    
    # 4. Regime-Specific Asymmetry Enhancement
    # Volatility Regime Classification
    range_avg_5 = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    data['range_expansion'] = (data['high'] - data['low']) / (range_avg_5 + eps)
    
    high_vol_condition = (data['range_expansion'] > 1.2) & (data['volume'] > volume_avg_5)
    low_vol_condition = (data['range_expansion'] < 0.8) & (data['volume'] < volume_avg_5)
    data['high_vol_regime'] = high_vol_condition.astype(int)
    data['low_vol_regime'] = low_vol_condition.astype(int)
    data['normal_regime'] = ((~high_vol_condition) & (~low_vol_condition)).astype(int)
    
    # Regime-Adaptive Signals
    data['high_vol_signal'] = data['asymmetry_momentum'] * data['distribution_alignment'] * data['volume_price_momentum_divergence']
    data['low_vol_signal'] = data['concentration_divergence'] * data['momentum_persistence'] * (1 / (data['range_expansion'] + eps))
    data['normal_signal'] = data['vol_asymmetry'] * data['volume_skewness'] * data['momentum_gap']
    
    # 5. Multi-Scale Convergence Framework
    # Timeframe Alignment Detection
    data['short_medium_alignment'] = np.sign(data['short_term_asymmetry']) * np.sign(data['medium_term_asymmetry'])
    data['volume_price_alignment'] = np.sign(data['volume_skewness']) * np.sign(data['price_distribution'])
    data['multi_scale_convergence'] = data['short_medium_alignment'] * data['volume_price_alignment']
    
    # Signal Reinforcement
    convergence_strength = []
    divergence_persistence = []
    
    for i in range(len(data)):
        if i < 3:
            convergence_strength.append(0)
        else:
            count = sum(data['multi_scale_convergence'].iloc[i-2:i+1] == 1)
            convergence_strength.append(count / 3)
        
        if i < 5:
            divergence_persistence.append(0)
        else:
            count = sum(data['volume_price_momentum_divergence'].iloc[i-4:i+1] > 0)
            divergence_persistence.append(count / 5)
    
    data['convergence_strength'] = convergence_strength
    data['divergence_persistence'] = divergence_persistence
    data['signal_quality'] = data['convergence_strength'] * data['divergence_persistence']
    
    # 6. Adaptive Weighting System
    # Volatility-Based Weights
    data['volatility_intensity'] = data['range_expansion']
    data['volume_intensity'] = data['volume_efficiency']
    data['combined_intensity'] = data['volatility_intensity'] * data['volume_intensity']
    
    # Persistence Adjustment
    signal_stability = []
    regime_stability = []
    
    for i in range(len(data)):
        if i < 3:
            signal_stability.append(0)
        else:
            count = sum(np.sign(data['asymmetry_momentum'].iloc[i-2:i+1]) == np.sign(data['concentration_divergence'].iloc[i-2:i+1]))
            signal_stability.append(count / 3)
        
        if i < 5:
            regime_stability.append(0)
        else:
            current_regime = []
            for j in range(i-4, i+1):
                if data['high_vol_regime'].iloc[j]:
                    current_regime.append('high')
                elif data['low_vol_regime'].iloc[j]:
                    current_regime.append('low')
                else:
                    current_regime.append('normal')
            count = sum(current_regime[k] == current_regime[k-1] for k in range(1, len(current_regime)))
            regime_stability.append(count / 4)
    
    data['signal_stability'] = signal_stability
    data['regime_stability'] = regime_stability
    data['stability_weight'] = data['signal_stability'] * data['regime_stability']
    
    # 7. Composite Alpha Construction
    # Regime-Specific Core
    data['high_vol_core'] = data['high_vol_signal'] * data['combined_intensity']
    data['low_vol_core'] = data['low_vol_signal'] * (1 / (data['combined_intensity'] + eps))
    data['normal_core'] = data['normal_signal'] * data['stability_weight']
    
    # Convergence Enhancement
    data['aligned_core'] = np.where(data['high_vol_regime'] == 1, data['high_vol_core'],
                                   np.where(data['low_vol_regime'] == 1, data['low_vol_core'], data['normal_core']))
    
    data['quality_adjusted'] = data['aligned_core'] * data['signal_quality']
    data['multi_scale_reinforced'] = data['quality_adjusted'] * data['multi_scale_convergence']
    
    # Final Alpha Synthesis
    data['base_alpha'] = data['multi_scale_reinforced'] * np.sign(data['vol_asymmetry'])
    
    # Enhanced Alpha
    enhanced_alpha = []
    for i in range(len(data)):
        if i < 3:
            enhanced_alpha.append(data['base_alpha'].iloc[i])
        else:
            count = sum(np.sign(data['base_alpha'].iloc[i-2:i+1]) == np.sign(data['asymmetry_momentum'].iloc[i-2:i+1]))
            enhanced_alpha.append(data['base_alpha'].iloc[i] * (count / 3))
    
    alpha_series = pd.Series(enhanced_alpha, index=data.index)
    
    return alpha_series
