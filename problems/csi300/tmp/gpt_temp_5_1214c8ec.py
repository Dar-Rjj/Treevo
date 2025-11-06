import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Efficiency
    data['short_term_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(20) - 1
    data['momentum_acceleration'] = (data['short_term_momentum'] - data['medium_term_momentum']) / np.where(data['short_term_momentum'] != 0, data['short_term_momentum'], np.nan)
    data['price_efficiency'] = (data['close'] - data['open']) / np.where((data['high'] - data['low']) != 0, (data['high'] - data['low']), np.nan)
    
    # Order Flow Microstructure Decay
    data['opening_order_flow_efficiency'] = (data['open'] - data['close'].shift(1)) * (data['amount'] / np.where(data['volume'] != 0, data['volume'], np.nan))
    data['closing_order_pressure'] = (data['close'] - (data['high'] + data['low'])/2) * ((data['amount'] - data['amount'].shift(1)) / np.where(data['amount'].shift(1) != 0, data['amount'].shift(1), np.nan))
    
    vol_avg_3 = (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3
    data['order_flow_decay_surge'] = data['volume'] / np.where(vol_avg_3 != 0, vol_avg_3, np.nan)
    
    data['order_concentration_decay'] = (data['amount'] / np.where(data['volume'] != 0, data['volume'], np.nan)) / np.where((data['amount'].shift(1) / np.where(data['volume'].shift(1) != 0, data['volume'].shift(1), np.nan)) != 0, 
                                                                 (data['amount'].shift(1) / np.where(data['volume'].shift(1) != 0, data['volume'].shift(1), np.nan)), np.nan)
    
    # Momentum-Microstructure Synthesis
    data['efficiency_momentum'] = data['momentum_acceleration'] * data['price_efficiency']
    data['order_flow_momentum'] = data['opening_order_flow_efficiency'] * data['closing_order_pressure']
    data['momentum_microstructure_alignment'] = data['efficiency_momentum'] * data['order_flow_momentum']
    
    # Multi-Scale Volatility Regimes
    data['short_term_volatility'] = (data['high'] - data['low']) / np.where((data['high'].shift(2) - data['low'].shift(2)) != 0, (data['high'].shift(2) - data['low'].shift(2)), np.nan)
    data['medium_term_volatility'] = (data['high'] - data['low']) / np.where((data['high'].shift(5) - data['low'].shift(5)) != 0, (data['high'].shift(5) - data['low'].shift(5)), np.nan)
    data['volatility_regime_strength'] = data['short_term_volatility'] * data['medium_term_volatility']
    
    daily_range_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            daily_range_persistence.iloc[i] = sum((data['high'].iloc[j] - data['low'].iloc[j]) / data['close'].iloc[j] for j in range(i-4, i+1))
    data['daily_range_persistence'] = daily_range_persistence
    
    # Volume-Amount Decay Patterns
    vol_avg_5 = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_momentum'] = data['volume'] / np.where(vol_avg_5 != 0, vol_avg_5, np.nan) - 1
    
    data['order_flow_concentration'] = (data['volume'] / np.where(data['amount'] != 0, data['amount'], np.nan)) - (data['volume'].shift(1) / np.where(data['amount'].shift(1) != 0, data['amount'].shift(1), np.nan))
    
    data['volume_order_divergence'] = (data['volume'] / np.where(data['volume'].shift(1) != 0, data['volume'].shift(1), np.nan) - 1) - (data['volume'] / np.where(data['volume'].shift(4) != 0, data['volume'].shift(4), np.nan) - 1)
    
    data['trade_size_momentum_decay'] = ((data['amount'] / np.where(data['volume'] != 0, data['volume'], np.nan)) / np.where((data['amount'].shift(1) / np.where(data['volume'].shift(1) != 0, data['volume'].shift(1), np.nan)) != 0, 
                                        (data['amount'].shift(1) / np.where(data['volume'].shift(1) != 0, data['volume'].shift(1), np.nan)), np.nan)) - 1
    
    # Regime-Volume Decay Integration
    data['volatility_volume_alignment'] = data['volatility_regime_strength'] * data['volume_momentum']
    data['order_flow_volume_decay'] = data['order_flow_concentration'] * data['volume_order_divergence']
    data['regime_order_flow_decay'] = data['volatility_volume_alignment'] * data['order_flow_volume_decay']
    
    # Asymmetry Decay Pattern Detection
    data['momentum_divergence'] = (data['close'] / np.where(data['close'].shift(1) != 0, data['close'].shift(1), np.nan) - 1) - (data['close'] / np.where(data['close'].shift(4) != 0, data['close'].shift(4), np.nan) - 1)
    
    data['volume_price_order_imbalance'] = ((data['close'] - data['close'].shift(1)) / np.where(data['close'].shift(1) != 0, data['close'].shift(1), np.nan)) - ((data['volume'] - data['volume'].shift(1)) / np.where(data['volume'].shift(1) != 0, data['volume'].shift(1), np.nan))
    
    data['efficiency_order_decay_divergence'] = (data['price_efficiency'] * (data['amount'] / np.where(data['volume'] != 0, data['volume'], np.nan))) - ((data['high'] - data['low']) * data['volume'] / np.where(data['amount'] != 0, data['amount'], np.nan))
    
    data['opening_closing_order_decay_asymmetry'] = data['opening_order_flow_efficiency'] - data['closing_order_pressure']
    
    # Multi-Scale Asymmetry Decay Integration
    data['momentum_volume_asymmetry_decay'] = data['momentum_divergence'] * data['volume_price_order_imbalance']
    data['efficiency_order_decay_asymmetry'] = data['efficiency_order_decay_divergence'] * data['opening_closing_order_decay_asymmetry']
    data['cross_asymmetry_decay_alignment'] = data['momentum_volume_asymmetry_decay'] * data['efficiency_order_decay_asymmetry']
    
    # Asymmetry Decay Persistence
    asymmetry_sign = np.sign(data['cross_asymmetry_decay_alignment'])
    asymmetry_duration = pd.Series(index=data.index, dtype=float)
    current_duration = 0
    current_sign = 0
    
    for i in range(len(data)):
        if i == 0 or np.isnan(asymmetry_sign.iloc[i]) or np.isnan(asymmetry_sign.iloc[i-1]):
            current_duration = 1
            current_sign = asymmetry_sign.iloc[i] if not np.isnan(asymmetry_sign.iloc[i]) else 0
        elif asymmetry_sign.iloc[i] == current_sign:
            current_duration += 1
        else:
            current_duration = 1
            current_sign = asymmetry_sign.iloc[i]
        asymmetry_duration.iloc[i] = current_duration
    
    data['asymmetry_decay_duration'] = asymmetry_duration
    data['asymmetry_decay_strength'] = data['cross_asymmetry_decay_alignment'] * data['asymmetry_decay_duration']
    data['persistent_asymmetry_decay_signal'] = data['asymmetry_decay_strength'] * data['daily_range_persistence']
    
    # Adaptive Decay Signal Synthesis
    data['volatility_decay_weight'] = 1 + np.abs(data['volatility_regime_strength'])
    data['order_decay_weight'] = 1 + np.abs(data['order_flow_concentration'])
    data['momentum_decay_weight'] = 1 + np.abs(data['momentum_acceleration'])
    data['asymmetry_decay_weight'] = 1 + np.abs(data['cross_asymmetry_decay_alignment'])
    
    # Dynamic Decay Signal Combination
    high_vol_regime = 0.5 * data['momentum_microstructure_alignment'] + 0.5 * data['regime_order_flow_decay']
    low_vol_regime = 0.7 * data['momentum_microstructure_alignment'] + 0.3 * data['regime_order_flow_decay']
    
    volatility_threshold = data['volatility_regime_strength'] > 1.0
    data['dynamic_decay_signal_combination'] = np.where(volatility_threshold, high_vol_regime, low_vol_regime)
    
    # Adaptive Decay Enhancement
    regime_weights = data['volatility_decay_weight'] * data['order_decay_weight'] * data['momentum_decay_weight'] * data['asymmetry_decay_weight']
    data['weighted_decay_signal'] = data['dynamic_decay_signal_combination'] * regime_weights
    data['asymmetry_decay_enhanced_signal'] = data['weighted_decay_signal'] * data['persistent_asymmetry_decay_signal']
    data['volume_confirmed_decay_signal'] = data['asymmetry_decay_enhanced_signal'] * data['trade_size_momentum_decay']
    
    # Composite Alpha Factor
    data['core_momentum_microstructure_decay'] = data['volume_confirmed_decay_signal'] * data['momentum_microstructure_alignment']
    data['volatility_regime_adjusted_decay'] = data['core_momentum_microstructure_decay'] * data['volatility_volume_alignment']
    data['asymmetry_decay_confirmed_output'] = data['volatility_regime_adjusted_decay'] * data['cross_asymmetry_decay_alignment']
    data['final_alpha'] = data['asymmetry_decay_confirmed_output'] * data['order_flow_volume_decay']
    
    # Return the final alpha factor series
    return data['final_alpha']
