import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['range'] = data['high'] - data['low']
    data['overnight_gap'] = data['open'] - data['close'].shift(1)
    data['intraday_move'] = data['close'] - data['open']
    data['avg_price'] = (data['high'] + data['low']) / 2
    data['midpoint_dev'] = abs(data['avg_price'] - (data['open'] + data['close']) / 2)
    data['trade_size'] = data['amount'] / data['volume']
    
    # Volatility Regime Detection
    # Recent Volatility Level
    data['vol_level'] = data['range'] / (data['range'].rolling(window=5).mean())
    
    # Volatility Trend
    data['vol_trend'] = data['range'] / data['range'].shift(4)
    
    # Volatility Clustering
    vol_increase = (data['range'] > data['range'].shift(1)).astype(int)
    data['vol_clustering'] = vol_increase.rolling(window=5).sum()
    
    # Regime Score
    data['regime_score'] = data['vol_level'] * data['vol_trend'] * data['vol_clustering']
    
    # Volume State Classification
    # Volume Intensity
    data['vol_intensity'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Volume Momentum
    data['vol_momentum'] = data['volume'] / data['volume'].shift(4)
    
    # Volume Stability
    data['vol_stability'] = 1 / (abs(data['volume'] - data['volume'].shift(1)) * data['volume'].shift(1)).replace(0, 1)
    
    # Volume State Score
    data['volume_state_score'] = data['vol_intensity'] * data['vol_momentum'] * data['vol_stability']
    
    # Market Microstructure State
    # Price-Volume Coherence
    data['pv_coherence'] = abs(data['close'] - data['open']) * data['volume'] / data['range'].replace(0, 1)
    
    # Trade Size Dynamics
    avg_trade_size = data['trade_size'].rolling(window=5).mean()
    data['trade_size_dynamics'] = data['trade_size'] / avg_trade_size.replace(0, 1)
    
    # Microstructure Efficiency
    data['micro_eff'] = abs(data['close'] - data['avg_price']) / data['range'].replace(0, 1)
    
    # State Composite
    data['state_composite'] = data['pv_coherence'] * data['trade_size_dynamics'] * data['micro_eff']
    
    # Adaptive Gap Analysis
    # Regime-Weighted Gap Components
    data['opening_gap_impact'] = data['overnight_gap'] * data['regime_score']
    data['intraday_gap_persistence'] = np.sign(data['overnight_gap']) * np.sign(data['intraday_move']) * data['volume_state_score']
    data['gap_absorption_rate'] = (abs(data['intraday_move']) / abs(data['overnight_gap']).replace(0, 1)) * data['state_composite']
    data['gap_momentum_eff'] = (data['intraday_move'] / abs(data['overnight_gap']).replace(0, 1)) * data['pv_coherence']
    
    # Multi-timeframe Gap Dynamics
    # Short-term Gap Memory (simplified correlation)
    gap_series = data['overnight_gap']
    gap_autocorr = gap_series.rolling(window=5).apply(lambda x: x.corr(x.shift(1)) if len(x) == 5 else 0, raw=False)
    data['short_term_gap_memory'] = gap_autocorr * data['regime_score']
    
    # Medium-term Gap Volatility
    data['medium_term_gap_vol'] = data['overnight_gap'].rolling(window=10).std() * data['volume_state_score']
    
    # Gap Amplitude Ratio
    data['gap_amplitude_ratio'] = abs(data['overnight_gap']) / abs(data['overnight_gap']).rolling(window=5).mean().replace(0, 1)
    
    # Gap Pattern Strength
    data['gap_pattern_strength'] = data['short_term_gap_memory'] * data['medium_term_gap_vol'] * data['gap_amplitude_ratio']
    
    # Volume-Enhanced Gap Signals
    data['volume_gap_alignment'] = data['gap_momentum_eff'] * data['vol_intensity']
    data['trade_flow_gap_confirmation'] = data['intraday_gap_persistence'] * data['trade_size_dynamics']
    data['microstructure_gap_eff'] = data['gap_absorption_rate'] * data['micro_eff']
    data['volume_gap_composite'] = data['volume_gap_alignment'] * data['trade_flow_gap_confirmation'] * data['microstructure_gap_eff']
    
    # Price Path Microstructure
    # Regime-Specific Path Characteristics
    data['path_directness'] = (abs(data['intraday_move']) / data['range'].replace(0, 1)) * data['regime_score']
    data['price_rejection_asymmetry'] = ((data['high'] - np.maximum(data['open'], data['close'])) / 
                                       (np.minimum(data['open'], data['close']) - data['low']).replace(0, 1)) * data['volume_state_score']
    data['midpoint_convergence'] = (1 - data['midpoint_dev'] / data['range'].replace(0, 1)) * data['state_composite']
    data['path_quality_score'] = data['path_directness'] * data['price_rejection_asymmetry'] * data['midpoint_convergence']
    
    # Multi-day Path Regime Adaptation
    data['overnight_gap_eff'] = (abs(data['overnight_gap']) / data['range'].replace(0, 1)) * data['regime_score']
    data['range_expansion_momentum'] = (data['range'] / data['range'].shift(4).replace(0, 1)) * data['volume_state_score']
    
    # Path Memory (simplified correlation)
    intraday_move = data['intraday_move']
    path_autocorr = intraday_move.rolling(window=5).apply(lambda x: x.corr(x.shift(1)) if len(x) == 5 else 0, raw=False)
    data['path_memory'] = path_autocorr * data['state_composite']
    
    data['adaptive_path_composite'] = data['overnight_gap_eff'] * data['range_expansion_momentum'] * data['path_memory']
    
    # Volume-Path Integration
    data['volume_path_coherence'] = data['path_quality_score'] * data['vol_intensity']
    data['trade_size_path_alignment'] = data['adaptive_path_composite'] * data['trade_size_dynamics']
    data['microstructure_path_eff'] = data['path_quality_score'] * data['micro_eff']
    data['volume_path_composite'] = data['volume_path_coherence'] * data['trade_size_path_alignment'] * data['microstructure_path_eff']
    
    # Dynamic Factor Integration
    # Regime-Adaptive Signal Combination
    data['gap_path_alignment'] = data['volume_gap_composite'] * data['volume_path_composite']
    data['vol_vol_regime_factor'] = data['regime_score'] * data['volume_state_score']
    data['microstructure_state_factor'] = data['state_composite'] * data['pv_coherence']
    data['dynamic_integration_core'] = data['gap_path_alignment'] * data['vol_vol_regime_factor'] * data['microstructure_state_factor']
    
    # Multi-timeframe Signal Synthesis
    data['short_term_component'] = data['gap_pattern_strength'] * data['path_quality_score']
    data['medium_term_component'] = data['medium_term_gap_vol'] * data['adaptive_path_composite']
    data['volume_confirmation'] = data['vol_intensity'] * data['trade_size_dynamics']
    data['timeframe_integrated_signal'] = data['short_term_component'] * data['medium_term_component'] * data['volume_confirmation']
    
    # Adaptive Weighting Mechanism
    data['regime_sensitivity_weight'] = data['regime_score'] * data['volume_state_score']
    data['microstructure_consistency_weight'] = data['state_composite'] * data['pv_coherence']
    data['gap_path_convergence_weight'] = data['gap_path_alignment'] * data['volume_path_composite']
    data['adaptive_weight_composite'] = data['regime_sensitivity_weight'] * data['microstructure_consistency_weight'] * data['gap_path_convergence_weight']
    
    # Final Alpha Construction
    # Core Alpha Signal
    data['dynamic_integration_signal'] = data['dynamic_integration_core'] * data['timeframe_integrated_signal']
    data['adaptive_weighting'] = data['dynamic_integration_signal'] * data['adaptive_weight_composite']
    data['volatility_regime_adjustment'] = data['adaptive_weighting'] / data['medium_term_gap_vol'].replace(0, 1)
    data['volume_confirmation_final'] = data['volatility_regime_adjustment'] * data['volume_confirmation']
    
    # Microstructure Refinement
    data['path_efficiency_enhancement'] = data['volume_confirmation_final'] * data['path_quality_score']
    data['gap_persistence_final'] = data['path_efficiency_enhancement'] * data['intraday_gap_persistence']
    data['trade_flow_alignment'] = data['gap_persistence_final'] * data['trade_size_dynamics']
    data['microstructure_alpha_core'] = data['trade_flow_alignment'] * data['micro_eff']
    
    # Regime-Adaptive Alpha Output
    data['regime_normalization'] = data['microstructure_alpha_core'] / data['regime_score'].replace(0, 1)
    data['volume_state_adjustment'] = data['regime_normalization'] * data['volume_state_score']
    data['final_adaptive_alpha'] = data['volume_state_adjustment'] * data['state_composite']
    
    # Return the final alpha factor
    return data['final_adaptive_alpha']
