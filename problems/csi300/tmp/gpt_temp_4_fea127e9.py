import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Regime-Adaptive Momentum with Efficiency-Volume Divergence
    """
    data = df.copy()
    
    # 1. Dynamic Multi-Frequency Regime Classification
    # Volatility Regime Analysis
    data['ret'] = data['close'].pct_change()
    data['range'] = data['high'] - data['low']
    
    # Multi-Timeframe Volatility Measurement
    data['vol_5d'] = data['ret'].rolling(5).std()
    data['vol_20d'] = data['ret'].rolling(20).std()
    data['range_vol_10d'] = (data['high'] / data['low']).rolling(10).std()
    data['vol_15d'] = data['ret'].rolling(15).std()
    
    # Volatility Momentum and Clustering
    data['vol_momentum'] = data['vol_5d'] / data['vol_5d'].shift(5)
    data['vol_cluster'] = data['vol_5d'] / data['vol_5d'].rolling(5).mean().shift(5)
    data['range_expansion'] = data['range'] / data['range'].shift(1)
    
    # Regime Identification
    high_vol_cond = (data['vol_5d'] > data['vol_20d'] * 1.8) & (data['vol_cluster'] > 1.2)
    low_vol_cond = (data['vol_5d'] < data['vol_20d'] * 0.6) & (data['vol_cluster'] < 0.8)
    data['vol_regime'] = 0  # Normal
    data.loc[high_vol_cond, 'vol_regime'] = 1  # High volatility
    data.loc[low_vol_cond, 'vol_regime'] = -1  # Low volatility
    
    # Efficiency-Volume Elasticity Framework
    data['range_util'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['efficiency_5d'] = (data['close'] - data['low'].shift(5)) / (data['high'].shift(5) - data['low'].shift(5))
    data['efficiency_persistence'] = data['range_util'].rolling(3).apply(lambda x: (x > 0.5).sum())
    
    # Volume Participation Dynamics
    data['volume_momentum'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) / (data['volume'].shift(1) / data['volume'].shift(2))
    data['volume_efficiency'] = data['volume'] / data['range']
    
    # Elasticity Classification
    data['efficiency_volume_corr'] = data['range_util'].rolling(5).corr(data['volume_momentum'])
    data['elasticity_momentum'] = data['efficiency_volume_corr'] / data['efficiency_volume_corr'].shift(3)
    data['elasticity_regime'] = np.where(data['efficiency_volume_corr'] > 0.3, 1, 
                                       np.where(data['efficiency_volume_corr'] < -0.3, -1, 0))
    
    # Regime Stability
    data['regime_stability'] = data['vol_regime'].rolling(3).apply(lambda x: (x == x.iloc[0]).sum() / 3)
    
    # 2. Multi-Timeframe Momentum-Efficiency Divergence
    # Cross-Frequency Momentum Structure
    data['ultra_short_mom'] = data['close'] / data['close'].shift(1) - 1
    data['short_term_mom'] = data['close'] / data['close'].shift(3) - 1
    data['medium_term_mom'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_curvature'] = (data['short_term_mom'] - data['medium_term_mom']) / abs(data['medium_term_mom'])
    
    # Efficiency-Momentum Integration
    data['efficiency_weighted_mom'] = data['range_util'] * data['ultra_short_mom']
    data['multi_scale_efficiency'] = data['efficiency_5d'] * data['efficiency_persistence']
    data['efficiency_momentum_div'] = data['efficiency_persistence'] * data['momentum_curvature']
    
    # Volume-Price Divergence Signals
    data['core_divergence'] = data['ultra_short_mom'] * data['volume_momentum'] * data['volume_acceleration']
    data['volume_confirmation'] = data['volume_momentum'] * data['range_expansion']
    data['multi_timeframe_align'] = np.sign(data['ultra_short_mom']) * np.sign(data['short_term_mom']) * np.sign(data['medium_term_mom'])
    data['divergence_persistence'] = data['core_divergence'].rolling(3).apply(lambda x: (x > 0).sum())
    
    # 3. Microstructure-Enhanced Breakout Framework
    data['liquidity_persistence'] = (data['amount'] > data['amount'].shift(1)).rolling(3).sum()
    data['volume_accel_confirmation'] = data['volume_acceleration'] * data['ultra_short_mom']
    
    # 4. Mean Reversion and Momentum Exhaustion
    data['short_term_reversion'] = -data['ultra_short_mom'].rolling(3).mean()
    data['efficiency_reversion'] = -data['efficiency_5d'].pct_change(3)
    data['momentum_exhaustion'] = -data['momentum_curvature']
    
    # Combined Mean Reversion Bias
    data['mean_reversion_bias'] = (data['short_term_reversion'] + data['efficiency_reversion'] + data['momentum_exhaustion']) / 3
    
    # 5. Composite Alpha Generation
    # Core Alpha Components
    core_alpha = (data['efficiency_momentum_div'] * data['core_divergence'] * 
                  data['volume_accel_confirmation'] * data['efficiency_persistence'])
    
    # Regime-Adaptive Weighting
    high_vol_weight = (data['core_divergence'] + data['volume_confirmation']) * (data['vol_regime'] == 1)
    low_vol_weight = (data['efficiency_momentum_div'] + data['multi_timeframe_align']) * (data['vol_regime'] == -1)
    high_elasticity_weight = (data['volume_confirmation'] + data['liquidity_persistence']) * (data['elasticity_regime'] == 1)
    transition_weight = (core_alpha * data['regime_stability']) * (data['vol_regime'] == 0)
    
    # Volatility-Context Adjustment
    vol_adjustment = np.where(data['vol_regime'] == 1, data['vol_momentum'],
                            np.where(data['vol_regime'] == -1, 1 / data['vol_5d'],
                                   data['regime_stability']))
    
    # Elasticity-Based Adjustment
    elasticity_adjustment = data['elasticity_momentum']
    
    # Final Composite Alpha
    regime_adaptive_composite = (core_alpha + high_vol_weight + low_vol_weight + 
                               high_elasticity_weight + transition_weight)
    
    final_alpha = (regime_adaptive_composite * vol_adjustment * elasticity_adjustment * 
                  (1 + data['mean_reversion_bias']) * data['divergence_persistence'])
    
    # Cross-Validation Filtering
    signal_confidence = (data['multi_timeframe_align'] + data['volume_confirmation'] + 
                        data['efficiency_persistence']) / 3
    
    alpha_output = final_alpha * signal_confidence
    
    return alpha_output
