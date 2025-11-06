import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Price-Volume Divergence Analysis
    data['price_momentum_5'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_10'] = data['close'] / data['close'].shift(10) - 1
    data['price_momentum_divergence'] = data['price_momentum_5'] - data['price_momentum_10']
    
    data['volume_momentum_5'] = data['volume'] / data['volume'].shift(5)
    data['volume_momentum_10'] = data['volume'] / data['volume'].shift(10)
    data['volume_momentum_divergence'] = data['volume_momentum_5'] - data['volume_momentum_10']
    
    data['price_volume_divergence_score'] = data['price_momentum_divergence'] * data['volume_momentum_divergence']
    
    data['divergence_persistence'] = (np.sign(data['price_volume_divergence_score']) * 
                                    (abs(data['price_volume_divergence_score'].shift(1)) + 
                                     abs(data['price_volume_divergence_score'].shift(2))))
    
    # Efficiency-Weighted Price Action
    data['opening_efficiency'] = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    data['efficiency_consistency'] = (np.sign(data['opening_efficiency']) * 
                                    np.sign(data['intraday_efficiency']) * 
                                    np.minimum(abs(data['opening_efficiency']), abs(data['intraday_efficiency'])))
    
    data['multi_timeframe_efficiency'] = data['efficiency_consistency'] * (data['close'] / data['close'].shift(3) - 1)
    
    # Volume Distribution Analysis
    data['volume_concentration'] = data['volume'] / (data['volume'] + data['volume'].shift(1) + 
                                                    data['volume'].shift(2) + data['volume'].shift(3))
    
    data['volume_spike_detection'] = data['volume'] / ((data['volume'].shift(1) + 
                                                       data['volume'].shift(2) + 
                                                       data['volume'].shift(3)) / 3)
    
    data['amount_per_volume'] = data['amount'] / data['volume']
    data['volume_quality_score'] = (data['volume_concentration'] * data['amount_per_volume'] * 
                                  (1 / data['volume_spike_detection']))
    
    # Range-Based Momentum Signals
    data['high_5d_rolling'] = data['high'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    data['low_5d_rolling'] = data['low'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].min() if len(x) > 1 else np.nan)
    
    data['range_breakout_strength'] = (data['close'] - data['high_5d_rolling']) / (data['high'] - data['low'])
    
    data['support_resistance_efficiency'] = ((data['close'] - data['low_5d_rolling']) / 
                                           (data['high_5d_rolling'] - data['low_5d_rolling']))
    
    data['range_compression'] = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))
    data['momentum_quality'] = (data['range_breakout_strength'] * data['support_resistance_efficiency'] * 
                              (1 / data['range_compression']))
    
    # Market Regime Classification
    data['volatility_regime'] = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))
    data['volume_regime'] = data['volume'] / ((data['volume'].shift(5) + data['volume'].shift(10)) / 2)
    
    data['high_volatility_regime'] = ((data['volatility_regime'] > 1.2) & (data['volume_regime'] > 1.1)).astype(int)
    data['low_volatility_regime'] = ((data['volatility_regime'] < 0.8) & (data['volume_regime'] < 0.9)).astype(int)
    data['normal_regime'] = ((~data['high_volatility_regime'].astype(bool)) & 
                           (~data['low_volatility_regime'].astype(bool))).astype(int)
    
    # Regime-Adaptive Factor Construction
    data['divergence_core'] = data['price_volume_divergence_score'] * data['divergence_persistence']
    data['volatility_enhanced_signal'] = data['divergence_core'] * data['volatility_regime']
    data['high_vol_alpha'] = data['volatility_enhanced_signal'] * data['momentum_quality']
    
    data['efficiency_core'] = data['efficiency_consistency'] * data['multi_timeframe_efficiency']
    data['volume_confirmed_signal'] = data['efficiency_core'] * data['volume_quality_score']
    data['low_vol_alpha'] = data['volume_confirmed_signal'] * data['range_breakout_strength']
    
    data['balanced_core'] = data['price_volume_divergence_score'] * data['efficiency_consistency']
    data['quality_weighted_signal'] = data['balanced_core'] * data['volume_quality_score']
    data['normal_alpha'] = data['quality_weighted_signal'] * data['momentum_quality']
    
    # Dynamic Factor Selection
    data['regime_alpha'] = np.where(data['high_volatility_regime'] == 1, data['high_vol_alpha'],
                                  np.where(data['low_volatility_regime'] == 1, data['low_vol_alpha'],
                                         data['normal_alpha']))
    
    # Composite Alpha Synthesis
    data['efficiency_weighted_alpha'] = data['regime_alpha'] * data['efficiency_consistency']
    data['final_factor'] = data['efficiency_weighted_alpha'] * data['volume_quality_score']
    
    # Return the final factor series
    return data['final_factor']
