import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Momentum Synchronization with Anchoring Effects
    """
    data = df.copy()
    
    # Volatility-Momentum Regime Classification
    # Multi-Scale Volatility Patterns
    data['true_range'] = data['high'] - data['low']
    data['true_range_expansion'] = data['true_range'] / data['true_range'].shift(1)
    data['asymmetric_vol_bias'] = (data['high'] - data['close']) / (data['close'] - data['low']).replace(0, np.nan)
    data['vol_regime_change'] = data['true_range'].rolling(5).mean() / data['true_range'].rolling(21).mean()
    
    # Momentum-Range Synchronization
    data['range_momentum_alignment'] = ((data['high']/data['close']) / (data['high'].shift(5)/data['close'].shift(5))) - \
                                      ((data['low']/data['close']) / (data['low'].shift(5)/data['close'].shift(5)))
    data['price_range_divergence'] = (data['close']/data['close'].shift(5)) - \
                                    (data['true_range']/data['true_range'].shift(5))
    data['multi_momentum_consistency'] = (data['close']/data['close'].shift(5)) * (data['close'].shift(5)/data['close'].shift(20))
    
    # Price Anchoring with Volume Efficiency
    # Anchoring Strength Analysis
    data['high_5d'] = data['high'].rolling(5).max()
    data['low_5d'] = data['low'].rolling(5).min()
    data['dist_from_5d_high'] = (data['high_5d'] - data['high']) / data['high']
    data['dist_from_5d_low'] = (data['low'] - data['low_5d']) / data['low']
    data['anchoring_pressure'] = data['dist_from_5d_high'] - data['dist_from_5d_low']
    
    # Volume-Efficient Range Utilization
    data['range_efficiency'] = data['true_range'] / (np.maximum(data['high'], data['close'].shift(1)) - np.minimum(data['low'], data['close'].shift(1)))
    data['volume_range_ratio'] = data['true_range'] / data['volume'].replace(0, np.nan)
    data['volume_trend_confirmation'] = data['volume'] / data['volume'].rolling(5).mean()
    
    # Asymmetric Volume-Momentum Flow
    # Directional Volume Pressure
    data['up_volume_intensity'] = np.where(data['close'] > data['close'].shift(1), data['volume'], 0)
    data['down_volume_intensity'] = np.where(data['close'] < data['close'].shift(1), data['volume'], 0)
    data['volume_pressure_asymmetry'] = (data['up_volume_intensity'] - data['down_volume_intensity']) / \
                                       (data['up_volume_intensity'] + data['down_volume_intensity']).replace(0, np.nan)
    
    # Momentum-Volume Synchronization
    data['volume_momentum_alignment'] = (data['close']/data['close'].shift(5)) * data['volume_trend_confirmation']
    data['momentum_accel_volume'] = ((data['close']/data['close'].shift(5)) - (data['close'].shift(5)/data['close'].shift(10))) * data['volume']
    
    # Flow persistence calculation
    data['volume_above_avg'] = data['volume'] > data['volume'].rolling(5).mean()
    data['flow_persistence'] = data['volume_above_avg'].rolling(10).apply(lambda x: len([i for i in range(1, len(x)) if x.iloc[i] and all(x.iloc[max(0, i-2):i+1])]), raw=False)
    
    # Amount-Based Microstructure Quality
    # Trade Size Efficiency
    data['large_trade_concentration'] = data['amount'] / data['volume'].replace(0, np.nan)
    
    # Trade size persistence calculation
    data['amount_above_avg'] = data['amount'] > data['amount'].rolling(5).mean()
    data['trade_size_persistence'] = data['amount_above_avg'].rolling(10).apply(lambda x: len([i for i in range(1, len(x)) if x.iloc[i] and all(x.iloc[max(0, i-2):i+1])]), raw=False)
    
    data['institutional_flow'] = (data['amount'] / data['amount'].rolling(5).mean()) * data['trade_size_persistence']
    
    # Price Impact Sensitivity
    data['amount_per_point'] = data['amount'] / data['true_range'].replace(0, np.nan)
    data['microstructure_efficiency'] = (data['close'] - data['close'].shift(1)) / data['amount_per_point'].replace(0, np.nan)
    data['quality_adjustment'] = data['microstructure_efficiency'] * data['large_trade_concentration']
    
    # Regime-Adaptive Signal Construction
    # High-Volatility Synchronization Signals
    data['breakout_momentum'] = data['dist_from_5d_high'] * data['true_range_expansion']
    data['vol_momentum_expansion'] = data['breakout_momentum'] * data['range_momentum_alignment']
    data['high_vol_flow_multiplier'] = data['vol_momentum_expansion'] * data['volume_pressure_asymmetry']
    
    # Low-Volatility Anchoring Signals
    data['mean_reversion_pressure'] = data['anchoring_pressure'] * data['range_efficiency']
    data['compression_release'] = data['mean_reversion_pressure'] * data['volume_range_ratio']
    data['low_vol_flow_adjustment'] = data['compression_release'] * data['volume_trend_confirmation']
    
    # Final Alpha Factor Generation
    # Multi-Timeframe Synchronization Blending
    data['vol_regime_selector'] = np.where(data['vol_regime_change'] > 1, 1, 0)
    data['regime_specific_sync'] = np.where(data['vol_regime_selector'] == 1, 
                                           data['high_vol_flow_multiplier'], 
                                           data['low_vol_flow_adjustment'])
    data['transition_smoothing'] = data['regime_specific_sync'] * (1 - abs(data['vol_regime_change'] - 1))
    
    # Microstructure Quality Enhancement
    data['flow_quality_multiplier'] = data['transition_smoothing'] * data['microstructure_efficiency']
    data['institutional_confirmation'] = data['flow_quality_multiplier'] * data['institutional_flow']
    
    # Output Composite Factor
    alpha_factor = data['institutional_confirmation']
    
    return alpha_factor
