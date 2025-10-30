import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Asymmetric Efficiency Framework
    data['bullish_efficiency'] = np.where(data['close'] > data['open'], 
                                        (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8), 
                                        0)
    data['bearish_efficiency'] = np.where(data['close'] < data['open'], 
                                        (data['open'] - data['close']) / (data['high'] - data['low'] + 1e-8), 
                                        0)
    data['asymmetry_ratio'] = data['bullish_efficiency'] / (data['bullish_efficiency'] + data['bearish_efficiency'] + 1e-8)
    
    # Multi-Timeframe Asymmetry
    data['rolling_bullish_bias'] = data['bullish_efficiency'].rolling(window=5).sum() / \
                                 (data['bullish_efficiency'].rolling(window=5).sum() + 
                                  data['bearish_efficiency'].rolling(window=5).sum() + 1e-8)
    data['asymmetry_momentum'] = data['asymmetry_ratio'] - data['asymmetry_ratio'].shift(3)
    data['directional_persistence'] = (data['bullish_efficiency'] > data['bearish_efficiency']).rolling(window=5).sum() / 5
    
    # Gap-Based Asymmetry
    data['gap_direction_efficiency'] = (abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)) * \
                                     np.sign(data['open'] - data['close'].shift(1))
    data['gap_fill_efficiency'] = (abs(data['close'] - data['close'].shift(1)) / (abs(data['open'] - data['close'].shift(1)) + 1e-8)) * \
                                np.sign(data['close'] - data['open'])
    data['gap_asymmetry_score'] = data['gap_direction_efficiency'] * data['gap_fill_efficiency']
    
    # Volume-Price Divergence System
    data['price_volume_divergence'] = ((data['close'] / data['close'].shift(1) - 1) * 
                                     (data['volume'] / data['volume'].shift(1) - 1))
    data['efficiency_volume_divergence'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * \
                                         (data['volume'] / data['volume'].shift(1) - 1)
    data['amount_price_divergence'] = ((data['close'] / data['close'].shift(1) - 1) * 
                                     (data['amount'] / data['amount'].shift(1) - 1))
    
    # Multi-Scale Divergence Analysis
    data['short_term_divergence'] = data['price_volume_divergence'] * np.sign(data['close'] - data['close'].shift(3))
    data['medium_term_divergence'] = data['efficiency_volume_divergence'] * np.sign(data['close'] - data['close'].shift(8))
    data['long_term_divergence'] = data['amount_price_divergence'] * np.sign(data['close'] - data['close'].shift(21))
    
    # Divergence Quality Assessment
    data['divergence_magnitude'] = abs(data['price_volume_divergence']) * abs(data['efficiency_volume_divergence'])
    
    def same_sign_count(x):
        signs = np.sign(x)
        return (signs == signs[0]).sum() if len(x) > 0 else 0
    
    data['divergence_consistency'] = pd.Series(
        [same_sign_count(data[['price_volume_divergence', 'efficiency_volume_divergence', 'amount_price_divergence']].iloc[i-2:i+1].values.flatten()) 
         if i >= 2 else 0 for i in range(len(data))], index=data.index) / 3
    
    data['divergence_persistence'] = (data['divergence_consistency'] > 0).rolling(window=5).sum() / 5
    
    # Microstructure Shadow Asymmetry
    data['upper_shadow_pressure'] = ((data['high'] - np.maximum(data['open'], data['close'])) / 
                                   (data['high'] - data['low'] + 1e-8)) * data['volume']
    data['lower_shadow_support'] = ((np.minimum(data['open'], data['close']) - data['low']) / 
                                  (data['high'] - data['low'] + 1e-8)) * data['volume']
    data['shadow_asymmetry_ratio'] = data['upper_shadow_pressure'] / (data['upper_shadow_pressure'] + data['lower_shadow_support'] + 1e-8)
    
    # Shadow-Volume Interaction
    data['pressure_volume_efficiency'] = data['upper_shadow_pressure'] * (data['volume'] / data['volume'].shift(1) - 1)
    data['support_volume_efficiency'] = data['lower_shadow_support'] * (data['volume'] / data['volume'].shift(1) - 1)
    data['shadow_flow_divergence'] = data['pressure_volume_efficiency'] - data['support_volume_efficiency']
    
    # Multi-Period Shadow Dynamics
    data['rolling_shadow_bias'] = data['shadow_asymmetry_ratio'].rolling(window=5).mean() * \
                                (data['volume'] / data['volume'].shift(1) - 1)
    data['shadow_momentum'] = data['shadow_asymmetry_ratio'] - data['shadow_asymmetry_ratio'].shift(3)
    data['shadow_stability'] = 1 / (data['shadow_asymmetry_ratio'].rolling(window=10).std() + 1e-8)
    
    # Price-Level Microstructure
    data['vwap_deviation'] = abs((data['high'] + data['low'] + data['close']) / 3 - (data['amount'] / data['volume'])) / \
                           (data['high'] - data['low'] + 1e-8)
    data['mid_point_efficiency'] = abs(data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'] + 1e-8)
    data['price_level_clustering'] = 1 / (data['amount'] / data['volume']).rolling(window=5).std().fillna(1e-8)
    
    # Microstructure Level Shifts
    data['level_break_efficiency'] = abs((data['amount'] / data['volume']) - (data['amount'].shift(1) / data['volume'].shift(1))) / \
                                   (data['high'] - data['low'] + 1e-8)
    data['level_momentum'] = (data['amount'] / data['volume']) / (data['amount'].shift(1) / data['volume'].shift(1)) - 1
    data['level_volume_alignment'] = data['level_momentum'] * (data['volume'] / data['volume'].shift(1) - 1)
    
    # Price-Level Quality
    data['level_consistency'] = 1 / (data['vwap_deviation'] + 1e-8)
    data['level_efficiency'] = abs(data['close'] - data['open']) / (data['level_break_efficiency'] + 1e-8)
    data['level_volume_quality'] = data['level_consistency'] * data['level_volume_alignment']
    
    # Momentum-Asymmetry Convergence
    data['bullish_momentum'] = (data['close'] / data['close'].shift(3) - 1) * data['bullish_efficiency']
    data['bearish_momentum'] = (data['close'] / data['close'].shift(3) - 1) * data['bearish_efficiency']
    data['net_asymmetric_momentum'] = data['bullish_momentum'] - data['bearish_momentum']
    
    # Divergence-Enhanced Momentum
    data['price_volume_momentum'] = (data['close'] / data['close'].shift(3) - 1) * data['price_volume_divergence']
    data['shadow_enhanced_momentum'] = data['net_asymmetric_momentum'] * data['shadow_flow_divergence']
    data['level_break_momentum'] = (data['close'] / data['close'].shift(3) - 1) * data['level_break_efficiency']
    
    # Convergence Quality Framework
    def multi_signal_alignment(row):
        signals = [row['net_asymmetric_momentum'], row['price_volume_momentum'], row['shadow_enhanced_momentum']]
        if len(signals) < 3:
            return 0
        signs = np.sign(signals)
        return (signs == signs[0]).sum() / 3
    
    data['multi_signal_alignment'] = data.apply(multi_signal_alignment, axis=1)
    data['convergence_strength'] = abs(data['net_asymmetric_momentum']) * abs(data['price_volume_momentum']) * abs(data['shadow_enhanced_momentum'])
    data['quality_score'] = data['convergence_strength'] * data['multi_signal_alignment']
    
    # Composite Alpha Construction
    # Core Asymmetry-Divergence Multiplier
    base_factor = data['net_asymmetric_momentum'] * data['price_volume_divergence']
    enhanced_factor = base_factor * data['shadow_flow_divergence'] * data['level_volume_alignment']
    scaled_factor = enhanced_factor * data['quality_score']
    
    # Additive Asymmetry Components
    shadow_asymmetry = data['shadow_asymmetry_ratio'] * data['shadow_momentum']
    volume_asymmetry = (data['bullish_efficiency'] - data['bearish_efficiency']) * (data['volume'] / data['volume'].shift(1) - 1)
    level_asymmetry = data['vwap_deviation'] * data['level_momentum']
    
    # Multi-Dimensional Validation
    asymmetry_divergence_alignment = data['net_asymmetric_momentum'] * data['price_volume_divergence'] * data['shadow_flow_divergence']
    volume_shadow_support = volume_asymmetry * shadow_asymmetry
    level_quality_validation = data['level_volume_quality'] * data['level_consistency']
    
    # Final composite alpha factor
    alpha_factor = (scaled_factor * 0.4 + 
                   shadow_asymmetry * 0.2 + 
                   volume_asymmetry * 0.15 + 
                   level_asymmetry * 0.1 + 
                   asymmetry_divergence_alignment * 0.08 + 
                   volume_shadow_support * 0.05 + 
                   level_quality_validation * 0.02)
    
    return alpha_factor
