import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volume Acceleration Patterns
    data['volume_ratio_1'] = data['volume'] / data['volume'].shift(1)
    data['volume_ratio_2'] = data['volume'].shift(1) / data['volume'].shift(2)
    data['volume_acceleration_ratio'] = data['volume_ratio_1'] / data['volume_ratio_2']
    data['acceleration_regime'] = np.sign(data['volume_acceleration_ratio'] - 1)
    
    # Acceleration Persistence
    acc_persistence = []
    for i in range(len(data)):
        if i < 2:
            acc_persistence.append(0)
        else:
            count = 0
            for j in range(max(0, i-1), max(0, i-3), -1):
                if j > 0 and data['acceleration_regime'].iloc[j] == data['acceleration_regime'].iloc[j-1]:
                    count += 1
            acc_persistence.append(count / 2)
    data['acceleration_persistence'] = acc_persistence
    
    # Volume Spike Analysis
    data['volume_spike_magnitude'] = data['volume_ratio_1'] - data['volume_ratio_2']
    data['spike_direction_alignment'] = np.sign(data['volume_spike_magnitude']) * np.sign(data['close'] - data['close'].shift(1))
    data['spike_regime'] = np.sign(data['volume_spike_magnitude']) * np.sign(data['volume_spike_magnitude'].shift(1))
    
    # Volume Distribution Characteristics
    data['volume_concentration'] = data['volume'] / (data['high'] - data['low'])
    data['concentration_change'] = data['volume_concentration'] - data['volume_concentration'].shift(1)
    data['volume_distribution_regime'] = np.sign(data['concentration_change']) * np.sign(data['concentration_change'].shift(1))
    
    # Asymmetric Price-Volume Response
    data['up_volume_impact'] = np.where(data['volume'] > data['volume'].shift(1), data['close'] - data['close'].shift(1), 0)
    data['down_volume_impact'] = np.where(data['volume'] < data['volume'].shift(1), data['close'] - data['close'].shift(1), 0)
    data['volume_impact_asymmetry'] = data['up_volume_impact'] - data['down_volume_impact']
    
    # Volume-Volatility Coupling
    data['volume_volatility_ratio'] = data['volume'] / (data['high'] - data['low'])
    data['volatility_adjusted_volume'] = data['volume_volatility_ratio'] * (data['close'] - data['close'].shift(1))
    data['coupling_asymmetry'] = np.where(data['volume'] > data['volume'].shift(1), data['volatility_adjusted_volume'], -data['volatility_adjusted_volume'])
    
    # Nonlinear Volume Response
    data['volume_response_curvature'] = (data['close'] - data['close'].shift(1)) * (data['volume_ratio_1'] ** 2)
    data['inverse_volume_response'] = (data['close'] - data['close'].shift(1)) / (data['volume_ratio_1'] ** 2)
    data['nonlinear_asymmetry'] = data['volume_response_curvature'] * data['inverse_volume_response']
    
    # Volume Regime Transitions
    data['volume_regime_change'] = np.sign(data['volume'] - data['volume'].shift(1)) * np.sign(data['volume'].shift(1) - data['volume'].shift(2))
    data['regime_transition_momentum'] = (data['close'] - data['close'].shift(1)) * data['volume_regime_change']
    
    # Regime Persistence Score
    regime_persistence = []
    for i in range(len(data)):
        if i < 2:
            regime_persistence.append(0)
        else:
            count = 0
            for j in range(max(0, i-1), max(0, i-3), -1):
                if j > 0 and np.sign(data['volume'].iloc[j] - data['volume'].iloc[j-1]) == np.sign(data['volume'].iloc[j-1] - data['volume'].iloc[j-2]):
                    count += 1
            regime_persistence.append(count / 2)
    data['regime_persistence_score'] = regime_persistence
    
    # Price-Volume Regime Alignment
    data['price_volume_coherence'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Coherence Persistence
    coherence_persistence = []
    for i in range(len(data)):
        if i < 2:
            coherence_persistence.append(0)
        else:
            count = 0
            for j in range(max(0, i-1), max(0, i-3), -1):
                if j > 0 and data['price_volume_coherence'].iloc[j] == data['price_volume_coherence'].iloc[j-1]:
                    count += 1
            coherence_persistence.append(count / 2)
    data['coherence_persistence'] = coherence_persistence
    data['aligned_regime_factor'] = data['price_volume_coherence'] * data['coherence_persistence']
    
    # Multi-Regime Integration
    data['volume_spike_regime'] = data['spike_regime'] * data['acceleration_regime']
    data['distribution_regime_coupling'] = data['volume_distribution_regime'] * data['volume_regime_change']
    data['integrated_transition'] = data['volume_spike_regime'] * data['distribution_regime_coupling']
    
    # Volume-Weighted Momentum
    data['volume_weighted_return'] = (data['close'] - data['close'].shift(1)) * data['volume']
    data['asymmetric_weighting'] = np.where(data['volume'] > data['volume'].shift(1), data['volume_weighted_return'], data['volume_weighted_return'] * 0.5)
    data['weighted_momentum_persistence'] = np.sign(data['asymmetric_weighting']) * np.sign(data['asymmetric_weighting'].shift(1))
    
    # Regime-Adaptive Momentum
    data['acceleration_based_momentum'] = (data['close'] - data['close'].shift(1)) * data['volume_acceleration_ratio']
    data['spike_enhanced_momentum'] = (data['close'] - data['close'].shift(1)) * data['volume_spike_magnitude']
    data['regime_momentum_composite'] = data['acceleration_based_momentum'] * data['spike_enhanced_momentum']
    
    # Asymmetric Signal Generation
    data['positive_asymmetry_signal'] = np.where(data['volume_impact_asymmetry'] > 0, data['volume_impact_asymmetry'], 0)
    data['negative_asymmetry_signal'] = np.where(data['volume_impact_asymmetry'] < 0, data['volume_impact_asymmetry'], 0)
    data['asymmetry_momentum'] = data['positive_asymmetry_signal'] - data['negative_asymmetry_signal']
    
    # Volume Regime Factors
    data['acceleration_regime_factor'] = data['acceleration_persistence'] * data['volume_acceleration_ratio']
    data['spike_regime_factor'] = data['spike_direction_alignment'] * data['spike_regime']
    data['distribution_regime_factor'] = data['volume_distribution_regime'] * data['volume_concentration']
    
    # Asymmetric Response Factors
    data['coupling_asymmetry_factor'] = data['coupling_asymmetry'] * data['volume_volatility_ratio']
    data['nonlinear_asymmetry_factor'] = data['nonlinear_asymmetry'] * data['volume_response_curvature']
    data['response_composite'] = data['coupling_asymmetry_factor'] * data['nonlinear_asymmetry_factor']
    
    # Integrated Dynamic Factors
    data['regime_response_alignment'] = data['acceleration_regime_factor'] * data['coupling_asymmetry_factor']
    data['spike_distribution_integration'] = data['spike_regime_factor'] * data['distribution_regime_factor']
    data['dynamic_factor_composite'] = data['regime_response_alignment'] * data['spike_distribution_integration']
    
    # Core Asymmetric Components
    data['volume_impact_asymmetry_enhanced'] = data['volume_impact_asymmetry'] * data['volume_regime_change']
    data['regime_momentum_asymmetry'] = data['regime_momentum_composite'] * data['asymmetry_momentum']
    data['dynamic_response_asymmetry'] = data['response_composite'] * data['dynamic_factor_composite']
    
    # Adaptive Weighting Scheme
    data['acceleration_weight'] = np.where(data['volume_acceleration_ratio'] > 1, 1.8, 1.2)
    data['spike_weight'] = np.where(data['volume_spike_magnitude'] > 0, 1.6, 1.4)
    data['distribution_weight'] = np.where(data['volume_distribution_regime'] > 0, 1.7, 1.3)
    
    # Weighted Signal Components
    data['weighted_impact'] = data['volume_impact_asymmetry_enhanced'] * data['acceleration_weight']
    data['weighted_regime'] = data['regime_momentum_asymmetry'] * data['spike_weight']
    data['weighted_response'] = data['dynamic_response_asymmetry'] * data['distribution_weight']
    
    # Final Alpha Construction
    data['primary_asymmetry'] = data['weighted_impact'] * data['regime_persistence_score']
    data['secondary_asymmetry'] = data['weighted_regime'] * data['aligned_regime_factor']
    data['tertiary_asymmetry'] = data['weighted_response'] * data['dynamic_factor_composite']
    
    # Composite Volume Asymmetry Alpha
    data['composite_volume_asymmetry_alpha'] = (
        data['primary_asymmetry'] * data['acceleration_weight'] +
        data['secondary_asymmetry'] * data['spike_weight'] +
        data['tertiary_asymmetry'] * data['distribution_weight']
    ) / (data['acceleration_weight'] + data['spike_weight'] + data['distribution_weight'])
    
    return data['composite_volume_asymmetry_alpha']
