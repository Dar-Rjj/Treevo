import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Momentum Dynamics
    data['short_term_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(8) - 1
    data['long_term_momentum'] = data['close'] / data['close'].shift(21) - 1
    
    # Momentum Acceleration Patterns
    data['short_term_acceleration'] = (data['short_term_momentum'] - data['medium_term_momentum']) / (np.abs(data['short_term_momentum']) + 1e-8)
    data['medium_term_acceleration'] = (data['medium_term_momentum'] - data['long_term_momentum']) / (np.abs(data['medium_term_momentum']) + 1e-8)
    
    # Momentum Coherence (5-day correlation)
    momentum_coherence = []
    for i in range(len(data)):
        if i >= 5:
            short_window = data['short_term_momentum'].iloc[i-4:i+1]
            medium_window = data['medium_term_momentum'].iloc[i-4:i+1]
            if len(short_window) == 5 and len(medium_window) == 5:
                corr = np.corrcoef(short_window, medium_window)[0,1]
                momentum_coherence.append(corr if not np.isnan(corr) else 0)
            else:
                momentum_coherence.append(0)
        else:
            momentum_coherence.append(0)
    data['momentum_coherence'] = momentum_coherence
    
    # Gap-Based Momentum Enhancement
    data['gap_momentum_efficiency'] = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)) * \
                                     ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8))
    
    gap_threshold = 0.5 * (data['high'].shift(1) - data['low'].shift(1))
    gap_persistence_count = []
    for i in range(len(data)):
        if i >= 1:
            count = 0
            for j in range(max(0, i-4), i+1):
                if j >= 1 and np.abs(data['open'].iloc[j] - data['close'].iloc[j-1]) > gap_threshold.iloc[j]:
                    count += 1
            gap_persistence_count.append(count)
        else:
            gap_persistence_count.append(0)
    data['gap_persistence'] = pd.Series(gap_persistence_count, index=data.index) * (data['amount'] / (data['volume'] + 1e-8))
    
    data['gap_amount_momentum'] = data['gap_momentum_efficiency'] * (data['amount'] / (data['volume'] + 1e-8)) / \
                                 (data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8) + 1e-8)
    
    # Price-Amount Microstructure Analysis
    data['opening_amount_efficiency'] = (data['open'] - data['close'].shift(1)) * (data['amount'] / (data['volume'] + 1e-8))
    data['opening_range_density'] = (data['high'] - data['low']) / (data['amount'] / (data['volume'] + 1e-8) + 1e-8)
    data['gap_amount_persistence'] = data['gap_persistence'] * (data['amount'] / (data['volume'] + 1e-8))
    
    data['realized_amount_efficiency'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * \
                                        (data['amount'] / (data['volume'] + 1e-8))
    data['closing_amount_pressure'] = (data['close'] - (data['high'] + data['low'])/2) * \
                                     (data['amount'] - data['amount'].shift(1)) / (data['amount'].shift(1) + 1e-8)
    data['volatility_amount_ratio'] = (data['high'] - data['low']) / (data['amount'] / (data['volume'] + 1e-8) + 1e-8)
    
    # Microstructure Asymmetry Patterns
    data['price_amount_asymmetry'] = ((data['high'] - data['open']) / (data['open'] - data['low'] + 1e-8)) * \
                                    ((data['amount'] / (data['volume'] + 1e-8)) / \
                                    (data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8) + 1e-8))
    data['up_move_amount_sensitivity'] = (data['high'] - data['open']) / (data['amount'] / (data['volume'] + 1e-8) + 1e-8)
    data['down_move_amount_sensitivity'] = (data['open'] - data['low']) / (data['amount'] / (data['volume'] + 1e-8) + 1e-8)
    data['directional_amount_asymmetry'] = data['up_move_amount_sensitivity'] / (data['down_move_amount_sensitivity'] + 1e-8)
    
    # Volume-Amount Flow Dynamics
    data['trade_size_momentum'] = (data['amount'] / (data['volume'] + 1e-8)) / \
                                 (data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8) + 1e-8) - 1
    data['volume_amount_efficiency'] = (data['close'] - data['open']) * data['volume'] / (data['amount'] + 1e-8)
    
    # Moving averages for flow concentration
    data['amount_ma5'] = data['amount'].rolling(window=5, min_periods=1).mean()
    data['volume_ma5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['flow_concentration'] = (data['amount'] / data['amount_ma5']) - (data['volume'] / data['volume_ma5'])
    
    # Microstructure Regime Detection
    avg_trade_size_prev = (data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8) + \
                          data['amount'].shift(2) / (data['volume'].shift(2) + 1e-8)) / 2
    data['large_trade_regime'] = (data['amount'] / (data['volume'] + 1e-8)) > (1.2 * avg_trade_size_prev)
    
    avg_volume_prev = (data['volume'].shift(1) + data['volume'].shift(2)) / 2
    avg_trade_size_prev2 = (data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8) + \
                           data['amount'].shift(2) / (data['volume'].shift(2) + 1e-8)) / 2
    data['small_trade_regime'] = (data['volume'] > 1.5 * avg_volume_prev) & \
                                ((data['amount'] / (data['volume'] + 1e-8)) < 0.8 * avg_trade_size_prev2)
    data['mixed_microstructure'] = ~data['large_trade_regime'] & ~data['small_trade_regime']
    
    # Flow Impact Analysis
    data['impact_efficiency'] = (data['close'] - data['open']) * data['volume'] / (data['high'] - data['low'] + 1e-8)
    data['amount_flow_divergence'] = data['flow_concentration'] * data['trade_size_momentum']
    data['microstructure_flow_momentum'] = data['impact_efficiency'] * data['trade_size_momentum']
    
    # Volatility-Amount Regime Adaptation
    data['short_term_volatility_break'] = (data['high'] - data['low']) / (data['high'].shift(2) - data['low'].shift(2) + 1e-8)
    data['medium_term_volatility_trend'] = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5) + 1e-8)
    data['volatility_regime_score'] = data['short_term_volatility_break'] * data['medium_term_volatility_trend']
    
    data['amount_surge_indicator'] = data['amount'] / ((data['amount'].shift(1) + data['amount'].shift(2) + data['amount'].shift(3)) / 3 + 1e-8)
    data['trade_size_regime'] = (data['amount'] / (data['volume'] + 1e-8)) / (data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8) + 1e-8)
    data['amount_efficiency_regime'] = ((data['close'] - data['open']) * data['amount']) / \
                                     ((data['close'].shift(1) - data['open'].shift(1)) * data['amount'].shift(1) + 1e-8)
    data['microstructure_transition'] = data['amount_surge_indicator'] * data['trade_size_regime']
    
    data['volatility_amount_breakout'] = data['volatility_regime_score'] * data['amount_surge_indicator']
    data['microstructure_regime_confirmation'] = data['trade_size_regime'] * data['amount_efficiency_regime']
    data['multi_regime_consistency'] = data['volatility_amount_breakout'] * data['microstructure_regime_confirmation']
    
    # Regime-Adaptive Signal Generation
    # High-Volatility Microstructure Regime
    data['volatility_amount_signal'] = data['volatility_amount_breakout'] * data['volume_amount_efficiency']
    data['momentum_enhanced_signal'] = data['gap_amount_momentum'] * data['short_term_acceleration']
    data['high_volatility_composite'] = data['volatility_amount_signal'] * data['momentum_enhanced_signal']
    
    # Low-Volatility Microstructure Regime
    data['amount_flow_signal'] = -data['amount_flow_divergence'] * data['flow_concentration']
    data['momentum_persistence_signal'] = data['medium_term_acceleration'] * data['momentum_coherence']
    data['low_volatility_composite'] = data['amount_flow_signal'] * data['momentum_persistence_signal']
    
    # Transition Microstructure Regime
    data['microstructure_momentum_signal'] = data['microstructure_flow_momentum'] * data['gap_momentum_efficiency']
    data['amount_asymmetry_signal'] = data['directional_amount_asymmetry'] * data['price_amount_asymmetry']
    data['transition_composite'] = data['microstructure_momentum_signal'] * data['amount_asymmetry_signal']
    
    # Final Alpha Synthesis
    # Core Signal Components
    data['momentum_amount_base'] = data['gap_amount_momentum'] * data['realized_amount_efficiency']
    data['volatility_flow_core'] = data['volatility_amount_ratio'] * data['microstructure_flow_momentum']
    data['asymmetry_momentum_core'] = data['directional_amount_asymmetry'] * data['short_term_acceleration']
    
    # Regime-Weighted Signals
    regime_weighted_signals = []
    for i in range(len(data)):
        if data['large_trade_regime'].iloc[i]:
            regime_weighted_signals.append(data['high_volatility_composite'].iloc[i])
        elif data['small_trade_regime'].iloc[i]:
            regime_weighted_signals.append(data['low_volatility_composite'].iloc[i])
        else:
            regime_weighted_signals.append(data['transition_composite'].iloc[i])
    data['regime_weighted_signals'] = pd.Series(regime_weighted_signals, index=data.index)
    
    # Final Alpha Factor
    data['momentum_coherence_scaling'] = data['regime_weighted_signals'] * data['momentum_coherence']
    data['multi_regime_enhancement'] = data['momentum_coherence_scaling'] * data['multi_regime_consistency']
    data['microstructure_enhanced_output'] = data['multi_regime_enhancement'] * data['microstructure_regime_confirmation']
    data['volatility_adapted_signal'] = data['microstructure_enhanced_output'] * data['volatility_regime_score']
    
    # Final factor combining all components
    alpha_factor = data['volatility_adapted_signal']
    
    return alpha_factor
