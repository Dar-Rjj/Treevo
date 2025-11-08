import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Price-Volume Efficiency with Order Flow Imbalance
    """
    data = df.copy()
    
    # Volatility Regime Classification
    # Daily range volatility
    data['daily_range_vol'] = (data['high'] - data['low']) / data['close']
    
    # Overnight volatility
    data['overnight_vol'] = np.abs(data['open'] / data['close'].shift(1) - 1)
    
    # Intraday efficiency
    data['intraday_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Composite volatility measure
    data['composite_vol'] = (data['daily_range_vol'] + data['overnight_vol'] + (1 - data['intraday_efficiency'])) / 3
    
    # Volatility persistence analysis
    data['vol_momentum_3d'] = data['composite_vol'] / data['composite_vol'].shift(3) - 1
    data['vol_regime_switch'] = (data['vol_momentum_3d'] > 0.1).astype(int)
    data['regime_stability'] = data['vol_regime_switch'].rolling(window=5, min_periods=3).sum()
    
    # Multi-scale volatility assessment
    data['vol_ratio_st_mt'] = data['composite_vol'].rolling(window=3, min_periods=2).mean() / \
                             data['composite_vol'].rolling(window=10, min_periods=7).mean()
    data['vol_compression'] = np.where(data['vol_ratio_st_mt'] < 0.8, 1.2, 1.0)
    data['vol_expansion'] = np.where(data['vol_ratio_st_mt'] > 1.2, 0.8, 1.0)
    
    # Order Flow Imbalance Measurement
    # Price-based imbalance proxies
    data['tick_directional'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['high_low_capture'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan) - 0.5
    data['gap_persistence'] = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['close'] - data['open'])
    
    # Volume-weighted imbalance signals
    data['volume_price_efficiency'] = data['volume'] * np.abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['cumulative_imbalance_3d'] = data['volume_price_efficiency'].rolling(window=3, min_periods=2).sum()
    data['imbalance_acceleration'] = data['volume_price_efficiency'] / data['volume_price_efficiency'].rolling(window=3, min_periods=2).mean()
    
    # Microstructure rejection filters
    vol_20d_p20 = data['volume'].rolling(window=20, min_periods=15).quantile(0.2)
    range_20d_p80 = data['daily_range_vol'].rolling(window=20, min_periods=15).quantile(0.8)
    
    data['low_volume_reject'] = (data['volume'] < vol_20d_p20).astype(int)
    data['extreme_vol_reject'] = (data['daily_range_vol'] > range_20d_p80).astype(int)
    data['gap_reversal'] = ((np.abs(data['open'] / data['close'].shift(1) - 1) > 0.02) & 
                           (np.sign(data['close'] - data['open']) != np.sign(data['open'] - data['close'].shift(1)))).astype(int)
    
    # Price-Volume Efficiency Scoring
    # Directional efficiency components
    data['price_movement_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_confirmation'] = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    data['combined_directional_score'] = data['price_movement_efficiency'] * data['volume_confirmation']
    
    # Regime-adaptive efficiency adjustment
    data['efficiency_high_vol_adj'] = data['combined_directional_score'] * data['vol_compression']
    data['efficiency_low_vol_adj'] = data['combined_directional_score'] * data['vol_expansion']
    data['regime_transition_penalty'] = np.where(data['vol_regime_switch'] == 1, 0.7, 1.0)
    
    # Efficiency persistence analysis
    data['efficiency_momentum_3d'] = data['combined_directional_score'] / data['combined_directional_score'].shift(3) - 1
    data['efficiency_mean_reversion'] = (data['combined_directional_score'] > data['combined_directional_score'].rolling(window=10, min_periods=7).quantile(0.8)).astype(int)
    
    # Multi-timeframe Order Flow Convergence
    # Short-term flow dynamics (1-3 days)
    data['flow_acceleration_st'] = data['imbalance_acceleration'].rolling(window=3, min_periods=2).mean()
    data['flow_direction_consistency'] = (np.sign(data['tick_directional']).rolling(window=3, min_periods=2).sum() / 3).abs()
    data['short_term_efficiency_val'] = data['combined_directional_score'].rolling(window=3, min_periods=2).mean()
    
    # Medium-term flow patterns (5-10 days)
    data['cumulative_imbalance_trend'] = data['cumulative_imbalance_3d'].rolling(window=10, min_periods=7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 7 else np.nan
    )
    data['medium_term_efficiency_conf'] = data['combined_directional_score'].rolling(window=10, min_periods=7).mean()
    
    # Timeframe alignment scoring
    data['timeframe_alignment'] = (np.sign(data['flow_acceleration_st']) == np.sign(data['cumulative_imbalance_trend'])).astype(float)
    data['efficiency_convergence'] = data['short_term_efficiency_val'] / data['medium_term_efficiency_conf']
    data['multi_scale_confirmation'] = data['timeframe_alignment'] * np.minimum(data['efficiency_convergence'], 2.0)
    
    # Adaptive Signal Construction
    # Core efficiency-imbalance engine
    data['raw_efficiency_imbalance'] = data['combined_directional_score'] * data['imbalance_acceleration']
    data['vol_regime_weight'] = np.where(data['composite_vol'] > data['composite_vol'].rolling(window=20, min_periods=15).median(), 
                                        data['vol_compression'], data['vol_expansion'])
    
    # Apply microstructure filters
    filter_mask = ~((data['low_volume_reject'] == 1) | (data['extreme_vol_reject'] == 1) | (data['gap_reversal'] == 1))
    data['filtered_signal'] = data['raw_efficiency_imbalance'] * filter_mask.astype(float)
    
    # Multi-timeframe convergence multiplier
    data['convergence_multiplier'] = (1 + data['multi_scale_confirmation'] * data['regime_transition_penalty'])
    
    # Regime-adaptive final signal
    data['vol_adjusted_alpha'] = data['filtered_signal'] * data['vol_regime_weight'] * data['convergence_multiplier']
    
    # Risk-Aware Signal Refinement
    # Extreme value protection
    vol_20d_median = data['composite_vol'].rolling(window=20, min_periods=15).median()
    data['position_size_adj'] = np.where(data['composite_vol'] > vol_20d_median * 1.5, 0.5, 1.0)
    
    # Flow divergence risk assessment
    data['flow_divergence_risk'] = (data['timeframe_alignment'] < 0.5).astype(float) * 0.7
    
    # Efficiency breakdown detection
    data['efficiency_breakdown'] = (data['combined_directional_score'] < data['combined_directional_score'].rolling(window=10, min_periods=7).quantile(0.3)).astype(float)
    
    # Regime transition management
    data['regime_change_signal'] = data['vol_regime_switch'].rolling(window=3, min_periods=2).sum()
    data['adaptive_attenuation'] = np.where(data['regime_change_signal'] >= 2, 0.6, 1.0)
    
    # Final Alpha Output
    data['final_alpha'] = (data['vol_adjusted_alpha'] * 
                          data['position_size_adj'] * 
                          (1 - data['flow_divergence_risk']) * 
                          (1 - data['efficiency_breakdown']) * 
                          data['adaptive_attenuation'])
    
    return data['final_alpha']
