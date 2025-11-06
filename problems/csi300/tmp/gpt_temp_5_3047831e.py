import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Asymmetric Volatility Framework
    # Dynamic Volatility Asymmetry
    data['range_expansion_asymmetry'] = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))) * np.sign(data['close'] - data['close'].shift(1))
    
    # Volatility Momentum with exponential weighting
    daily_range = data['high'] - data['low']
    vol_momentum = daily_range.rolling(window=10, min_periods=5).std()
    data['volatility_momentum'] = vol_momentum.ewm(span=5, adjust=False).mean()
    data['volatility_adjusted_asymmetry'] = data['range_expansion_asymmetry'] / data['volatility_momentum']
    
    # Price Momentum Asymmetry
    data['short_term_bias'] = (data['close'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Medium-term skew with 5-day average high-low range
    high_avg_5d = data['high'].rolling(window=5, min_periods=3).mean()
    low_avg_5d = data['low'].rolling(window=5, min_periods=3).mean()
    data['medium_term_skew'] = (data['close'] - data['close'].shift(5)) / (high_avg_5d - low_avg_5d)
    data['acceleration_asymmetry'] = data['short_term_bias'] - data['medium_term_skew']
    
    # Volume-Volatility Coupling
    data['volume_price_asymmetry'] = ((data['volume'] / data['volume'].shift(1) - 1) * np.sign(data['close'] - data['close'].shift(1)))
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5, min_periods=3).sum()
    data['volume_volatility_asymmetry'] = data['volume_price_asymmetry'] * data['volatility_adjusted_asymmetry']
    
    # Microstructure Regime Identification
    # Liquidity Proxy Components
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    data['volume_concentration_asymmetry'] = ((data['amount'] / data['volume']) / (data['amount'].shift(1) / data['volume'].shift(1))) * np.sign(data['close'] - data['close'].shift(1))
    data['price_impact'] = (data['close'] - data['open']) * data['volume']
    
    # Multi-Dimensional Regime Classification
    data['regime'] = 'transition'
    data.loc[(data['spread_proxy'] < 0.02) & (data['volume_concentration'] < 0.4), 'regime'] = 'high_liquidity'
    data.loc[(data['spread_proxy'] > 0.05) | (data['volume_concentration'] > 0.7), 'regime'] = 'low_liquidity'
    
    # Regime Persistence Assessment
    data['volatility_state'] = ((data['high'] - data['low']) / data['close']) / ((data['high'].shift(5) - data['low'].shift(5)) / data['close'].shift(5))
    
    # Momentum state (sum of price direction signs over 5 days)
    momentum_signs = pd.Series(np.zeros(len(data)), index=data.index)
    for i in range(5):
        momentum_signs += np.sign(data['close'] - data['close'].shift(i+1))
    data['momentum_state'] = momentum_signs
    
    # Regime confidence (count consistent regime signals over past 3 periods)
    regime_consistency = pd.Series(np.zeros(len(data)), index=data.index)
    for i in range(3):
        regime_consistency += (data['regime'] == data['regime'].shift(i+1)).astype(int)
    data['regime_confidence'] = regime_consistency
    
    # Microstructure Asymmetry Components
    # Price-Microstructure Dynamics
    data['price_impact_asymmetry'] = (abs(data['close'] - data['close'].shift(1)) / data['amount']) * np.sign(data['short_term_bias'])
    data['efficiency_microstructure'] = ((data['high'] - data['low']) / data['amount']) * np.sign(data['medium_term_skew'])
    
    intraday_bias = abs(((data['high'] + data['low']) / 2 - (data['open'] + data['close']) / 2) / (data['high'] - data['low']))
    data['intraday_bias_asymmetry'] = intraday_bias * np.sign(data['short_term_bias'])
    
    # Volume-Microstructure Analysis
    data['volume_efficiency_asymmetry'] = (data['volume'] / data['amount']) * np.sign(data['volume_price_asymmetry'])
    data['volume_fractal'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(1) / data['volume'].shift(2))
    data['microstructure_persistence'] = (data['amount'] / data['amount'].shift(1)) * np.sign(data['volume_concentration_asymmetry'])
    
    # Fractal Order Flow
    data['price_volume_coupling'] = ((data['close'] / data['close'].shift(1)) * (data['volume'] / data['volume'].shift(1))) - ((data['close'].shift(1) / data['close'].shift(2)) * (data['volume'].shift(1) / data['volume'].shift(2)))
    data['volume_weighted_acceleration'] = ((data['close'] / data['close'].shift(1) - 1) * data['volume']) - ((data['close'].shift(1) / data['close'].shift(2) - 1) * data['volume'].shift(1))
    
    # Fractal Consistency (5-day correlation)
    fractal_consistency = pd.Series(np.zeros(len(data)), index=data.index)
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            if len(window_data) >= 3:
                corr = window_data['volume_fractal'].corr(window_data['price_volume_coupling'])
                fractal_consistency.iloc[i] = corr if not np.isnan(corr) else 0
    data['fractal_consistency'] = fractal_consistency
    
    # Asymmetry Convergence Framework
    # Price-Volatility Convergence
    data['short_term_convergence'] = data['short_term_bias'] * data['intraday_bias_asymmetry']
    data['medium_term_alignment'] = data['medium_term_skew'] * data['efficiency_microstructure']
    data['volatility_momentum_integration'] = data['acceleration_asymmetry'] * data['volatility_adjusted_asymmetry']
    
    # Volume-Microstructure Synthesis
    data['volume_distribution_convergence'] = data['volume_price_asymmetry'] * data['volume_efficiency_asymmetry']
    data['amount_efficiency_integration'] = data['volume_concentration_asymmetry'] * data['microstructure_persistence']
    data['fractal_microstructure'] = data['volume_fractal'] * data['price_volume_coupling']
    
    # Cross-Dimensional Asymmetry
    data['multi_scale_convergence'] = data['short_term_convergence'] * data['medium_term_alignment']
    data['microstructure_volatility_convergence'] = data['volume_efficiency_asymmetry'] * data['volatility_adjusted_asymmetry']
    data['regime_adaptive_convergence'] = data['multi_scale_convergence'] * data['regime_confidence']
    
    # Regime-Adaptive Signal Construction
    # Base Asymmetry Signal
    base_signal = pd.Series(np.zeros(len(data)), index=data.index)
    high_liquidity_mask = data['regime'] == 'high_liquidity'
    low_liquidity_mask = data['regime'] == 'low_liquidity'
    transition_mask = data['regime'] == 'transition'
    
    base_signal[high_liquidity_mask] = (data['volatility_momentum_integration'] * data['price_impact'] * data['volume_weighted_acceleration'])[high_liquidity_mask]
    base_signal[low_liquidity_mask] = (data['volatility_momentum_integration'].shift(1) * data['price_impact'].shift(1) * data['volume_weighted_acceleration'].shift(1))[low_liquidity_mask]
    
    # For transition regime, average all asymmetry components
    asymmetry_components = [data['volatility_momentum_integration'], data['price_impact'], data['volume_weighted_acceleration'],
                          data['short_term_convergence'], data['medium_term_alignment'], data['volume_distribution_convergence']]
    transition_avg = sum(comp[transition_mask] for comp in asymmetry_components) / len(asymmetry_components)
    base_signal[transition_mask] = transition_avg
    
    # Microstructure Confirmation System
    data['triple_confirmation'] = data['price_impact_asymmetry'] + data['volume_efficiency_asymmetry'] + data['efficiency_microstructure']
    data['confirmation_intensity'] = (abs(data['triple_confirmation']) / 3) * data['regime_confidence']
    data['fractal_enhancement'] = base_signal * data['fractal_consistency']
    
    # Convergence Quality Assessment
    data['asymmetry_regime_coherence'] = np.sign(data['acceleration_asymmetry']) * np.sign(data['momentum_state'])
    
    # Quality Multiplier
    quality_multiplier = pd.Series(1.0, index=data.index)
    high_quality_mask = (data['fractal_consistency'] > 0.8) & (data['asymmetry_regime_coherence'] > 0)
    medium_quality_mask = ((data['fractal_consistency'] >= 0.5) & (data['fractal_consistency'] <= 0.8)) | (data['asymmetry_regime_coherence'] == 0)
    low_quality_mask = (data['fractal_consistency'] < 0.3) | (data['asymmetry_regime_coherence'] < 0)
    
    quality_multiplier[high_quality_mask] = 2.0
    quality_multiplier[medium_quality_mask] = 1.5
    quality_multiplier[low_quality_mask] = 0.8
    
    # Final Alpha Assembly
    # Enhanced Convergence Signal
    data['microstructure_confirmed_signal'] = data['fractal_enhancement'] * data['confirmation_intensity']
    data['quality_enhanced_convergence'] = data['microstructure_confirmed_signal'] * quality_multiplier
    
    # Risk-Asymmetry Adjustment
    data['mean_reversion_component'] = (data['close'] - data['close'].rolling(window=20, min_periods=10).mean()) / (data['high'] - data['low'])
    data['volatility_scaling'] = data['quality_enhanced_convergence'] / ((data['high'] - data['low']) / data['close'])
    data['asymmetry_stability_blend'] = data['volatility_scaling'] * 0.7 + data['mean_reversion_component'] * 0.3
    
    # Final Factor Output
    data['regime_adaptive_final'] = data['asymmetry_stability_blend'] * data['volume_efficiency_asymmetry']
    final_alpha = data['regime_adaptive_final'] * np.sign(data['acceleration_asymmetry'])
    
    return final_alpha
