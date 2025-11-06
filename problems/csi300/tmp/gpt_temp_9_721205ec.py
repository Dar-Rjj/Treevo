import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Cross-Scale Momentum Dynamics
    # Relative Momentum Framework
    data['short_term_momentum'] = data['close'] / data['close'].shift(3) - data['close'].shift(1) / data['close'].shift(4)
    data['medium_term_momentum'] = data['close'] / data['close'].shift(10) - data['close'].shift(1) / data['close'].shift(11)
    data['momentum_acceleration'] = data['short_term_momentum'] - data['medium_term_momentum']
    data['volatility_enhanced_momentum'] = data['momentum_acceleration'] * (data['high'].shift(1) - data['low'].shift(1))
    
    # Fractal Range Analysis
    data['range'] = data['high'] - data['low']
    data['short_term_range_avg'] = data['range'].rolling(window=5).mean()
    data['medium_term_range_avg'] = data['range'].rolling(window=20).mean()
    data['short_term_range_scaling'] = data['range'] / data['short_term_range_avg']
    data['medium_term_range_scaling'] = data['range'] / data['medium_term_range_avg']
    data['range_fractality_ratio'] = data['short_term_range_scaling'] / data['medium_term_range_scaling']
    data['volatility_weighted_fractality'] = data['range_fractality_ratio'] * (data['high'].shift(1) - data['low'].shift(1))
    
    # Momentum-Range Integration
    data['volatility_momentum_alignment'] = data['volatility_enhanced_momentum'] * data['range_fractality_ratio']
    data['fractal_momentum_signal'] = data['volatility_momentum_alignment'] * data['volatility_weighted_fractality']
    data['multi_scale_momentum_core'] = data['fractal_momentum_signal'] * data['momentum_acceleration']
    
    # Dynamic Volatility Microstructure
    # Opening-Closing Dynamics
    data['volatility_adjusted_gap_momentum'] = (data['open'] - data['close'].shift(1)) * (data['open'] - data['open'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['volatility_adjusted_closing_momentum'] = (data['close'] - data['open']) * (data['close'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['opening_closing_asymmetry'] = data['volatility_adjusted_gap_momentum'] - data['volatility_adjusted_closing_momentum']
    data['range_efficiency'] = (data['close'] - data['open']) * data['volume'] / (data['amount'] * (data['high'] - data['low']))
    
    # Volume-Volatility Integration
    data['volatility_weighted_volume_momentum'] = data['volume'] * (data['volume'] - data['volume'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['range_based_volume_decay'] = data['volume'] / data['volume'].shift(1) - data['range'] / data['range'].shift(1)
    data['volume_range_correlation'] = (data['volume'] / data['volume'].shift(1)) * (data['close'] - data['open']) / (data['high'].shift(1) - data['low'].shift(1))
    data['microstructure_volatility'] = data['volume'] * (data['close'] - data['open']) / (data['amount'] * data['range'])
    
    # Volatility Microstructure Patterns
    data['volatility_decay_divergence'] = data['range_based_volume_decay'] - data['microstructure_volatility']
    data['range_volume_decay'] = data['range_based_volume_decay'] * data['volatility_decay_divergence']
    data['opening_range_efficiency'] = (data['open'] - data['close'].shift(1)) / data['range']
    data['closing_volatility_efficiency'] = (data['close'] - data['open']) / data['range']
    data['volatility_efficiency_alignment'] = data['opening_range_efficiency'] * data['closing_volatility_efficiency']
    data['microstructure_momentum'] = data['volume_range_correlation'] * data['microstructure_volatility']
    
    # Asymmetry Pattern Detection
    # Order Flow Asymmetry
    data['vwap'] = data['amount'] / data['volume']
    data['opening_order_flow'] = (data['open'] - data['close'].shift(1)) * data['vwap']
    data['closing_order_pressure'] = (data['close'] - (data['high'] + data['low'])/2) * (data['amount'] - data['amount'].shift(1)) / data['amount'].shift(1)
    data['order_flow_asymmetry'] = data['opening_order_flow'] - data['closing_order_pressure']
    data['volume_adjusted_asymmetry'] = data['order_flow_asymmetry'] * (data['volume'] / data['volume'].shift(1))
    
    # Range Asymmetry Patterns
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['up_day_mask'] = data['price_change'] > 0
    data['down_day_mask'] = data['price_change'] < 0
    
    def rolling_up_range(series):
        return series.rolling(window=5).apply(lambda x: x[data['up_day_mask'].loc[x.index]].mean(), raw=False)
    
    def rolling_down_range(series):
        return series.rolling(window=5).apply(lambda x: x[data['down_day_mask'].loc[x.index]].mean(), raw=False)
    
    data['up_day_range_avg'] = rolling_up_range(data['range'])
    data['down_day_range_avg'] = rolling_down_range(data['range'])
    data['range_asymmetry'] = (data['range'] / data['up_day_range_avg']) / (data['range'] / data['down_day_range_avg'])
    data['volatility_enhanced_asymmetry'] = data['range_asymmetry'] * (data['high'].shift(1) - data['low'].shift(1))
    
    # Asymmetry Integration
    data['flow_range_alignment'] = data['volume_adjusted_asymmetry'] * data['range_asymmetry']
    data['volatility_weighted_asymmetry'] = data['flow_range_alignment'] * (data['high'].shift(1) - data['low'].shift(1))
    data['cross_asymmetry_signal'] = data['volatility_weighted_asymmetry'] * data['order_flow_asymmetry']
    
    # Dynamic Regime Adaptation
    # Volatility Regime Framework
    data['normalized_range'] = data['range'] / data['close']
    data['short_term_volatility'] = data['normalized_range'].rolling(window=5).mean()
    data['medium_term_volatility'] = data['normalized_range'].rolling(window=20).mean()
    data['volatility_regime'] = data['short_term_volatility'] / data['medium_term_volatility']
    
    data['range_above_avg'] = data['range'] > data['range'].rolling(window=5).mean()
    data['volatility_persistence'] = data['range_above_avg'].rolling(window=5, min_periods=1).apply(lambda x: (x == True).sum(), raw=False)
    
    # Flow Regime Classification
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['amount_ratio'] = data['amount'] / data['amount'].rolling(window=5).mean()
    data['high_flow_indicator'] = ((data['volume_ratio'] > 1.2) & (data['amount_ratio'] > 1.1)).astype(int)
    data['low_flow_indicator'] = ((data['volume_ratio'] < 0.8) & (data['amount_ratio'] < 0.9)).astype(int)
    data['flow_regime'] = data['high_flow_indicator'] - data['low_flow_indicator']
    
    # Dynamic Regime Integration
    data['volatility_adaptive_momentum'] = data['multi_scale_momentum_core'] * data['volatility_regime']
    data['flow_enhanced_microstructure'] = data['microstructure_momentum'] * data['flow_regime']
    data['regime_weighted_asymmetry'] = data['cross_asymmetry_signal'] * data['volatility_persistence']
    data['dynamic_core_factor'] = data['volatility_adaptive_momentum'] * data['flow_enhanced_microstructure'] * data['regime_weighted_asymmetry']
    
    # Efficiency and Momentum Decay
    # Price Efficiency Metrics
    data['gap_efficiency'] = ((data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))) * ((data['close'] - data['open']) / data['range'])
    data['price_impact'] = ((data['close'] - data['open']) / data['range']) * data['vwap']
    data['flow_consistency'] = np.sign(data['volume_ratio']) * np.sign(data['amount_ratio'])
    data['efficiency_composite'] = data['gap_efficiency'] * data['price_impact'] * data['flow_consistency']
    
    # Momentum Decay Patterns
    data['volatility_momentum_decay'] = data['microstructure_volatility'] * data['range_based_volume_decay']
    data['range_efficiency_decay'] = data['opening_closing_asymmetry'] * data['range_efficiency']
    data['volume_momentum_decay'] = data['volatility_weighted_volume_momentum'] * data['volume_range_correlation']
    data['decay_composite'] = data['volatility_momentum_decay'] * data['range_efficiency_decay'] * data['volume_momentum_decay']
    
    # Efficiency-Decay Integration
    data['efficiency_weighted_decay'] = data['decay_composite'] * data['efficiency_composite']
    data['momentum_decay_alignment'] = data['efficiency_weighted_decay'] * data['momentum_acceleration']
    data['dynamic_efficiency_core'] = data['momentum_decay_alignment'] * data['flow_consistency']
    
    # Composite Alpha Factor
    # Core Integration
    data['momentum_volatility_base'] = data['dynamic_core_factor'] * data['multi_scale_momentum_core']
    data['asymmetry_enhanced_core'] = data['momentum_volatility_base'] * data['cross_asymmetry_signal']
    data['efficiency_weighted_core'] = data['asymmetry_enhanced_core'] * data['dynamic_efficiency_core']
    
    # Dynamic Enhancement
    data['volatility_confirmed_signal'] = data['efficiency_weighted_core'] * data['volatility_regime']
    data['flow_adaptive_enhancement'] = data['volatility_confirmed_signal'] * data['flow_regime']
    data['range_weighted_output'] = data['flow_adaptive_enhancement'] * data['range_fractality_ratio']
    
    # Final Alpha
    data['multi_scale_momentum_volatility_asymmetry'] = data['range_weighted_output'] * data['opening_closing_asymmetry']
    data['microstructure_enhanced_factor'] = data['multi_scale_momentum_volatility_asymmetry'] * data['microstructure_momentum']
    data['efficiency_confirmed_output'] = data['microstructure_enhanced_factor'] * data['efficiency_composite']
    
    alpha = data['efficiency_confirmed_output']
    
    return alpha
