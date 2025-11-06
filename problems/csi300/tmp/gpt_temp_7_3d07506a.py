import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['hl_range'] = data['high'] - data['low']
    data['amount_per_volume'] = data['amount'] / (data['volume'] + 1e-8)  # Avoid division by zero
    data['hl_over_apv'] = data['hl_range'] / (data['amount_per_volume'] + 1e-8)
    
    # Quantum Entropy Momentum components
    data['opening_entropy_momentum'] = (data['open'] - data['prev_close']) * data['volume'] * (data['hl_over_apv'] ** (1/3))
    data['intraday_entropy_momentum'] = ((data['high'] + data['low'])/2 - data['open']) * data['volume'] * (data['hl_over_apv'] ** (1/3))
    data['closing_entropy_momentum'] = (data['close'] - (data['high'] + data['low'])/2) * data['volume'] * (data['hl_over_apv'] ** (1/3))
    
    # Entropy Pattern Recognition
    data['short_term_entropy_resonance'] = data['closing_entropy_momentum'].rolling(window=3).sum() / (data['closing_entropy_momentum'].rolling(window=7).sum() + 1e-8)
    data['gap_entropy_dynamics'] = np.sign(data['open'] - data['prev_close']) * np.sign(data['close'] - data['open']) * data['hl_over_apv']
    data['volume_entropy_alignment'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['prev_volume']) * data['hl_over_apv']
    
    # Entropy Regime Detection
    data['entropy_volatility_regime'] = (data['hl_over_apv'] ** 0.5) / (data['hl_over_apv'] ** 0.5).rolling(window=9).mean()
    data['entropy_flow_regime'] = data['closing_entropy_momentum'].rolling(window=9).std()
    data['liquidity_efficiency_entropy'] = np.abs(data['close'] - data['open']) / (data['hl_over_apv'] + 1e-8)
    
    # Fractal Entropy Analysis
    data['micro_entropy_efficiency'] = data['close'].diff().abs().rolling(window=2).sum() / data['hl_range'].rolling(window=2).sum()
    data['macro_entropy_efficiency'] = data['close'].diff().abs().rolling(window=8).sum() / data['hl_range'].rolling(window=8).sum()
    data['fractal_entropy_ratio'] = data['micro_entropy_efficiency'] / (data['macro_entropy_efficiency'] + 1e-8)
    
    # Volume-Liquidity Entropy
    data['volume_acceleration'] = data['volume'] / (data['prev_volume'] + 1e-8) - 1
    data['liquidity_intensity'] = data['amount'] / (data['hl_over_apv'] + 1e-8)
    
    # Calculate rolling correlations for Volume-Liquidity Coherence
    volume_liquidity_corr = []
    for i in range(len(data)):
        if i >= 10:
            window_data = data.iloc[i-9:i+1]
            corr = window_data['volume'].corr(window_data['hl_over_apv'])
            volume_liquidity_corr.append(corr if not np.isnan(corr) else 0)
        else:
            volume_liquidity_corr.append(0)
    
    data['volume_liquidity_coherence'] = volume_liquidity_corr
    data['volume_liquidity_coherence_shift'] = data['volume_liquidity_coherence'].shift(5)
    data['volume_liquidity_coherence_diff'] = data['volume_liquidity_coherence'] - data['volume_liquidity_coherence_shift']
    
    # Entropy Alpha Combination
    data['short_term_entropy_factor'] = data['short_term_entropy_resonance'] * data['gap_entropy_dynamics'] * data['fractal_entropy_ratio']
    
    # Medium-term entropy factor
    data['volume_hl_ratio'] = data['volume'] / (data['hl_over_apv'] ** (2/3) + 1e-8)
    data['volume_hl_ratio_4'] = data['volume_hl_ratio'].shift(4)
    data['medium_term_entropy_factor'] = data['volume_entropy_alignment'] * (data['volume_hl_ratio'] - data['volume_hl_ratio_4'])
    
    # Adaptive Entropy Alpha
    volatility_threshold_high = data['entropy_volatility_regime'].quantile(0.7)
    volatility_threshold_low = data['entropy_volatility_regime'].quantile(0.3)
    
    conditions = [
        data['entropy_volatility_regime'] > volatility_threshold_high,
        data['entropy_volatility_regime'] < volatility_threshold_low
    ]
    choices = [
        data['short_term_entropy_factor'] * 0.7 + data['medium_term_entropy_factor'] * 0.3,
        data['short_term_entropy_factor'] * 0.3 + data['medium_term_entropy_factor'] * 0.7
    ]
    data['adaptive_entropy_alpha'] = np.select(conditions, choices, default=data['short_term_entropy_factor'] * 0.5 + data['medium_term_entropy_factor'] * 0.5)
    
    # Entropy Regime Integration
    data['volatility_regime_strength'] = np.abs(data['entropy_volatility_regime'] - 1)
    data['efficiency_regime_strength'] = data['liquidity_efficiency_entropy']
    data['coherence_regime_strength'] = data['volume_liquidity_coherence_diff'] * data['fractal_entropy_ratio']
    
    # Regime-specific enhancement
    efficiency_threshold = data['liquidity_efficiency_entropy'].quantile(0.7)
    coherence_threshold = data['volume_liquidity_coherence_diff'].quantile(0.7)
    
    enhancement_conditions = [
        (data['liquidity_efficiency_entropy'] > efficiency_threshold) & (data['entropy_volatility_regime'] > volatility_threshold_high),
        (data['liquidity_efficiency_entropy'] > efficiency_threshold) & (data['volume_liquidity_coherence_diff'] > coherence_threshold),
        (data['liquidity_efficiency_entropy'] < data['liquidity_efficiency_entropy'].quantile(0.3)) & (data['entropy_volatility_regime'] > volatility_threshold_high)
    ]
    enhancement_choices = [
        1.2 * data['efficiency_regime_strength'],
        1.3 * data['coherence_regime_strength'],
        1.1 * data['volatility_regime_strength']
    ]
    data['regime_enhancement'] = np.select(enhancement_conditions, enhancement_choices, default=1.0)
    
    data['enhanced_entropy_alpha'] = data['adaptive_entropy_alpha'] * data['regime_enhancement']
    
    # Final Entropy Alpha
    data['entropy_momentum_alignment'] = np.sign(data['closing_entropy_momentum']) * np.sign(data['volume_acceleration'])
    data['volume_efficiency_consistency'] = data['volume_liquidity_coherence_diff'] * data['liquidity_efficiency_entropy']
    data['entropy_confirmation_strength'] = data['entropy_momentum_alignment'] * data['volume_efficiency_consistency']
    
    # Dynamic Optimization
    data['efficiency_boost'] = 1 + (data['liquidity_efficiency_entropy'] * 0.4)
    data['coherence_multiplier'] = 1 + (data['volume_liquidity_coherence_diff'] * 0.3)
    data['volatility_adjustment'] = 1 + (data['volatility_regime_strength'] * 0.2)
    
    # Final Alpha
    data['final_alpha'] = (data['enhanced_entropy_alpha'] * 
                          data['entropy_confirmation_strength'] * 
                          data['efficiency_boost'] * 
                          data['coherence_multiplier'] * 
                          data['volatility_adjustment'])
    
    return data['final_alpha']
