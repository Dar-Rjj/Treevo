import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize all intermediate calculations
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    
    # Price Fracture Efficiency components
    data['gap_fracture_efficiency'] = ((data['open'] - data['prev_close']) / 
                                     (data['prev_high'] - data['prev_low'] + 1e-8) * 
                                     ((data['amount']/data['volume'])/(data['prev_amount']/data['prev_volume'] + 1e-8)))
    
    data['intraday_fracture_efficiency'] = ((data['close'] - data['open']) / 
                                          (data['high'] - data['low'] + 1e-8) * 
                                          (data['volume'] / data['volume'].shift(5).replace(0, 1e-8)))
    
    # Fracture Persistence Efficiency
    price_fracture = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    fracture_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        window = price_fracture.iloc[i-5:i+1]
        fracture_persistence.iloc[i] = (window > 0.02).sum() * ((data['close'].iloc[i] - data['close'].iloc[i-1]) / 
                                                               (data['high'].iloc[i] - data['low'].iloc[i] + 1e-8))
    data['fracture_persistence_efficiency'] = fracture_persistence
    
    data['fracture_rejection_efficiency'] = ((abs(data['close'] - data['high']) - abs(data['close'] - data['low'])) * 
                                           (data['volume'] / data['prev_volume'].replace(0, 1e-8)))
    
    # Volume-Efficiency Fracture Dynamics
    data['volume_fracture_efficiency'] = ((data['volume'] / data['prev_volume'].replace(0, 1e-8) - 1) * 
                                        ((data['close'] - data['open']) / data['open'].replace(0, 1e-8)))
    
    data['efficiency_concentration_fracture'] = ((data['volume'] / data['amount'].replace(0, 1e-8)) * 
                                               ((data['high'] - data['low']) / data['prev_close'].replace(0, 1e-8)))
    
    data['volume_price_fracture_alignment'] = (np.sign(data['volume_fracture_efficiency']) * 
                                             np.sign(data['gap_fracture_efficiency']))
    
    data['efficiency_momentum_fracture'] = ((data['volume'] / data['volume'].shift(5).replace(0, 1e-8) - 1) * 
                                          ((data['amount']/data['volume'])/(data['prev_amount']/data['prev_volume'] + 1e-8)))
    
    # Microstructure Fracture-Efficiency Integration
    price_fracture_efficiency = data['gap_fracture_efficiency'] + data['intraday_fracture_efficiency']
    fracture_efficiency_corr = pd.Series(index=data.index, dtype=float)
    for i in range(3, len(data)):
        window_price = price_fracture_efficiency.iloc[i-2:i+1]
        window_volume = data['volume_fracture_efficiency'].iloc[i-2:i+1]
        if len(window_price) >= 2 and len(window_volume) >= 2:
            fracture_efficiency_corr.iloc[i] = window_price.corr(window_volume)
    data['fracture_efficiency_correlation'] = fracture_efficiency_corr.fillna(0)
    
    fracture_momentum_efficiency = pd.Series(index=data.index, dtype=float)
    for i in range(2, len(data)):
        window_sum = 0
        for j in range(i-2, i+1):
            window_sum += price_fracture_efficiency.iloc[j] * data['volume_fracture_efficiency'].iloc[j]
        fracture_momentum_efficiency.iloc[i] = window_sum
    data['fracture_momentum_efficiency'] = fracture_momentum_efficiency
    
    data['microstructure_health_score'] = 1 - abs(data['fracture_efficiency_correlation'])
    
    # Volatility-Liquidity Regime Detection
    data['volatility_clustering_ratio'] = ((data['high'] - data['low']) / 
                                         ((data['high'].rolling(5).mean() - data['low'].rolling(5).mean()) + 1e-8))
    
    regime_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(9, len(data)):
        count = 0
        for j in range(i-9, i+1):
            if j > 0 and j-1 > 0:
                ratio1 = data['close'].iloc[j] / data['close'].iloc[j-1]
                ratio2 = data['close'].iloc[j-1] / data['close'].iloc[j-2]
                if ratio1 > ratio2:
                    count += 1
        regime_persistence.iloc[i] = count / 10
    data['regime_persistence'] = regime_persistence
    
    data['volatility_fracture_alignment'] = data['volatility_clustering_ratio'] * data['gap_fracture_efficiency']
    
    # Liquidity Dynamics Integration
    data['bid_ask_spread_proxy'] = ((data['high'] - data['low']) / 
                                  (data['amount']/data['volume'].replace(0, 1e-8) + 1e-8))
    
    data['liquidity_momentum'] = ((data['amount']/data['volume']) / 
                                (data['amount'].shift(5)/data['volume'].shift(5).replace(0, 1e-8) + 1e-8) - 1)
    
    data['liquidity_efficiency_fracture'] = data['bid_ask_spread_proxy'] * data['volume_fracture_efficiency']
    
    # Price Efficiency Regimes
    data['overnight_gap_efficiency'] = ((data['open']/data['prev_close'] - 1) / 
                                      ((data['high'] - data['low'])/data['prev_close'] + 1e-8))
    
    data['intraday_return_consistency'] = ((data['close']/data['open'] - 1) - 
                                         (data['close'].shift(1)/data['open'].shift(1) - 1))
    
    data['efficiency_regime_score'] = data['overnight_gap_efficiency'] * data['intraday_return_consistency']
    
    # Regime-Switching Fracture Dynamics
    data['volatility_driven_fracture'] = data['gap_fracture_efficiency'] * data['volatility_clustering_ratio']
    data['high_volatility_efficiency'] = data['volume_fracture_efficiency'] * data['volatility_driven_fracture']
    
    volatility_regime_count = pd.Series(index=data.index, dtype=float)
    for i in range(3, len(data)):
        window = data['volatility_clustering_ratio'].iloc[i-2:i+1]
        volatility_regime_count.iloc[i] = (window > 1.2).sum() * (data['volume'].iloc[i] / data['volume'].iloc[i-10] if i >= 10 else 1)
    data['volatility_regime_score'] = volatility_regime_count
    
    data['consolidation_breakout_potential'] = ((data['high'] - data['low']) / 
                                              (data['high'].rolling(5).mean() - data['low'].rolling(5).mean() + 1e-8))
    
    data['liquidity_accumulation'] = np.where(data['consolidation_breakout_potential'] < 0.7,
                                            data['volume'] / data['prev_volume'].replace(0, 1e-8), 1)
    
    data['low_volatility_quality'] = (1 - abs(data['consolidation_breakout_potential'] - 1)) * data['liquidity_momentum']
    
    # Transition Regime Dynamics
    data['regime_change_detection'] = np.sign(data['volatility_clustering_ratio'] - 
                                            data['volatility_clustering_ratio'].rolling(5).mean())
    
    data['transition_efficiency'] = ((data['close'] - data['prev_close']) * 
                                   data['regime_change_detection'] * data['liquidity_momentum'])
    
    data['transition_quality'] = (abs(data['transition_efficiency']) / (data['high'] - data['low'] + 1e-8) * 
                                (data['volume'] / data['prev_volume'].replace(0, 1e-8)))
    
    # Multi-Scale Integration Framework
    data['immediate_fracture_momentum'] = data['gap_fracture_efficiency'] * data['volume_fracture_efficiency']
    data['liquidity_confirmation'] = data['immediate_fracture_momentum'] * data['liquidity_momentum']
    data['short_term_convergence'] = data['liquidity_confirmation'] * data['fracture_persistence_efficiency']
    
    data['weekly_volatility_trend'] = (data['volatility_clustering_ratio'].rolling(5).mean() / 
                                     data['volatility_clustering_ratio'].shift(5).rolling(5).mean().replace(0, 1e-8))
    
    data['efficiency_trend_alignment'] = (data['overnight_gap_efficiency'].rolling(5).mean() / 
                                        data['overnight_gap_efficiency'].shift(5).rolling(5).mean().replace(0, 1e-8))
    
    data['medium_term_convergence'] = (data['weekly_volatility_trend'] * data['efficiency_trend_alignment'] * 
                                     data['liquidity_momentum'])
    
    scale_regime_alignment = pd.Series(index=data.index, dtype=float)
    for i in range(5, len(data)):
        window_short = data['short_term_convergence'].iloc[i-4:i+1]
        window_medium = data['medium_term_convergence'].iloc[i-4:i+1]
        if len(window_short) >= 2 and len(window_medium) >= 2:
            scale_regime_alignment.iloc[i] = window_short.corr(window_medium)
    data['scale_regime_alignment'] = scale_regime_alignment.fillna(0)
    
    data['regime_health_score'] = data['microstructure_health_score'] * data['transition_quality']
    data['integrated_regime_signal'] = (data['scale_regime_alignment'] * data['regime_health_score'] * 
                                      (data['volume'] / data['volume'].shift(10).replace(0, 1e-8)))
    
    # Dynamic Anchoring Enhancement
    data['gap_capture_fracture'] = (((data['close'] - data['open']) / (abs(data['open'] - data['prev_close']) + 1e-8)) * 
                                  np.sign(data['open'] - data['prev_close']))
    
    data['gap_fade_detection'] = (np.sign(data['open'] - data['prev_close']) * 
                                np.sign(data['close'] - data['open']))
    
    data['opening_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    data['volatility_utilization'] = (data['high'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    data['liquidity_momentum_fracture'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8) * 
                                         data['liquidity_momentum'])
    
    data['session_quality'] = data['gap_capture_fracture'] * data['opening_efficiency']
    
    data['efficiency_weighted'] = data['session_quality'] * (data['volume'] / (data['high'] - data['low'] + 1e-8))
    data['liquidity_confirmed'] = data['efficiency_weighted'] * data['liquidity_momentum']
    data['regime_adaptive'] = (data['liquidity_confirmed'] * 
                             ((data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3) + 1e-8) - 1))
    
    # Adaptive Regime Enhancement
    data['high_volatility_amplifier'] = 1 + data['volatility_regime_score']
    data['low_volatility_amplifier'] = 1 + data['low_volatility_quality']
    data['transition_amplifier'] = 1 + abs(data['transition_quality'])
    
    data['high_volatility_boost'] = (data['high_volatility_efficiency'] * data['high_volatility_amplifier'] * 
                                   (data['volume'] / data['prev_volume'].replace(0, 1e-8)))
    
    data['low_volatility_breakout'] = (data['liquidity_accumulation'] * data['low_volatility_amplifier'] * 
                                     data['liquidity_momentum'])
    
    data['transition_magnification'] = (data['transition_efficiency'] * data['transition_amplifier'] * 
                                      (data['volume'] / data['volume'].shift(5).replace(0, 1e-8)))
    
    # Fracture-Volatility Regime Detection
    def get_regime(row):
        if row['volatility_clustering_ratio'] > 1.2:
            return 'high'
        elif row['volatility_clustering_ratio'] < 0.8:
            return 'low'
        else:
            return 'transition'
    
    data['fracture_volatility_regime'] = data.apply(get_regime, axis=1)
    
    # Adaptive Signal Routing
    data['adaptive_signal'] = np.where(data['fracture_volatility_regime'] == 'high', data['high_volatility_boost'],
                                     np.where(data['fracture_volatility_regime'] == 'low', data['low_volatility_breakout'],
                                             data['transition_magnification']))
    
    # Composite Alpha Construction
    data['high_volatility_core'] = data['high_volatility_boost'] * data['volatility_regime_score']
    data['low_volatility_core'] = data['low_volatility_breakout'] * data['low_volatility_quality']
    data['transition_core'] = data['transition_magnification'] * data['transition_quality']
    
    data['regime_weighted_core'] = (data['high_volatility_core'] * data['high_volatility_amplifier'] +
                                  data['low_volatility_core'] * data['low_volatility_amplifier'] +
                                  data['transition_core'] * data['transition_amplifier'])
    
    data['fracture_liquidity_enhancement'] = data['regime_weighted_core'] * data['integrated_regime_signal']
    data['momentum_validation'] = data['fracture_liquidity_enhancement'] * data['fracture_momentum_efficiency']
    
    # Final Alpha Factor
    data['anchoring_enhanced_base'] = data['momentum_validation'] * data['regime_adaptive']
    data['quality_enhanced_output'] = data['anchoring_enhanced_base'] * data['regime_health_score']
    data['final_alpha'] = data['quality_enhanced_output'] * data['microstructure_health_score']
    
    return data['final_alpha']
