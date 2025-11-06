import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility-Regime Momentum with Flow Asymmetry alpha factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Identification
    # Current Volatility
    data['current_vol'] = (data['high'] - data['low']) / data['close']
    
    # Historical Volatility (5-day MA of 5-day lagged volatility)
    data['hist_vol'] = ((data['high'].shift(5) - data['low'].shift(5)) / data['close'].shift(5)).rolling(window=5).mean()
    data['vol_regime'] = data['current_vol'] / data['hist_vol']
    
    # Regime Classification
    conditions = [
        data['vol_regime'] > 2.0,
        (data['vol_regime'] >= 1.0) & (data['vol_regime'] <= 2.0),
        data['vol_regime'] < 1.0
    ]
    choices = ['high', 'medium', 'low']
    data['regime_class'] = np.select(conditions, choices, default='medium')
    
    # Regime Persistence
    regime_changes = data['regime_class'] != data['regime_class'].shift(1)
    data['regime_change_flag'] = regime_changes.astype(int)
    
    # Consecutive days in current regime
    data['consecutive_days'] = 0
    for i in range(1, len(data)):
        if data['regime_class'].iloc[i] == data['regime_class'].iloc[i-1]:
            data.loc[data.index[i], 'consecutive_days'] = data['consecutive_days'].iloc[i-1] + 1
    
    data['regime_stability'] = data['consecutive_days'] / 8
    
    # Recent regime change (days since last change)
    data['days_since_change'] = 0
    last_change_idx = 0
    for i in range(1, len(data)):
        if data['regime_change_flag'].iloc[i] == 1:
            last_change_idx = i
        data.loc[data.index[i], 'days_since_change'] = i - last_change_idx
    
    # Multi-Timeframe Momentum Analysis
    # Intraday Momentum Components
    data['bullish_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['bearish_pressure'] = (data['open'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    data['net_intraday_bias'] = data['bullish_pressure'] - data['bearish_pressure']
    
    # Multi-Period Momentum
    data['immediate_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['short_term_momentum'] = data['close'] / data['close'].shift(4) - 1
    data['momentum_acceleration'] = data['immediate_momentum'] / data['short_term_momentum'].replace(0, np.nan)
    
    # Gap and Overnight Momentum
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_capture'] = (data['close'] - data['open']) / data['open']
    data['total_momentum_efficiency'] = data['overnight_gap'] * data['intraday_capture']
    
    # Volume-Flow Confirmation System
    # Volume Momentum Patterns
    data['volume_trend'] = data['volume'] / data['volume'].shift(4)
    data['price_volume_alignment'] = np.sign(data['volume_trend'] - 1) * np.sign(data['short_term_momentum'])
    data['volume_intensity'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Amount-Based Flow Analysis
    data['trade_size_momentum'] = (data['amount'] / data['volume']) / (data['amount'].shift(4) / data['volume'].shift(4)).replace(0, np.nan)
    data['price_size_correlation'] = data['immediate_momentum'] * (data['amount'] / data['volume'])
    data['flow_efficiency'] = data['trade_size_momentum'] * data['price_size_correlation']
    
    # Intraday Flow Asymmetry
    data['opening_flow'] = (data['open'] - data['close'].shift(1)) * data['volume']
    data['closing_flow'] = (data['close'] - data['open']) * data['volume']
    data['flow_asymmetry'] = data['opening_flow'] / data['closing_flow'].replace(0, np.nan)
    
    # Range Efficiency and Compression
    # Volatility-Adjusted Returns
    data['daily_efficiency'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Multi-Day Efficiency
    range_sum = sum((data['high'].shift(i) - data['low'].shift(i)) for i in range(5))
    data['multi_day_efficiency'] = (data['close'] - data['close'].shift(4)) / range_sum.replace(0, np.nan)
    data['efficiency_momentum'] = data['daily_efficiency'] / data['multi_day_efficiency'].replace(0, np.nan)
    
    # Compression Detection
    data['range_compression'] = (data['high'] - data['low']) / (data['high'] - data['low']).rolling(window=5).mean()
    data['volume_compression'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['compression_phase'] = (data['range_compression'] < 0.8) & (data['volume_compression'] < 0.8)
    
    # Breakout Probability
    data['compression_duration'] = 0
    current_duration = 0
    for i in range(len(data)):
        if data['compression_phase'].iloc[i]:
            current_duration += 1
        else:
            current_duration = 0
        data.loc[data.index[i], 'compression_duration'] = current_duration
    
    # Cross-Timeframe Divergence Detection
    # Price-Momentum Divergence
    data['short_vs_medium_divergence'] = data['immediate_momentum'] - data['short_term_momentum']
    data['gap_vs_intraday_divergence'] = data['overnight_gap'] - data['intraday_capture']
    data['momentum_acceleration_divergence'] = data['momentum_acceleration'] - 1
    
    # Volume-Price Divergence
    data['volume_momentum_divergence'] = data['volume_trend'] - data['short_term_momentum']
    data['flow_momentum_divergence'] = data['flow_asymmetry'] * data['net_intraday_bias']
    data['efficiency_divergence'] = data['daily_efficiency'] - data['multi_day_efficiency']
    
    # Regime-Momentum Alignment
    high_vol_mask = data['regime_class'] == 'high'
    low_vol_mask = data['regime_class'] == 'low'
    
    data['high_vol_alignment'] = data['net_intraday_bias'] * high_vol_mask.astype(int)
    data['low_vol_alignment'] = data['momentum_acceleration'] * low_vol_mask.astype(int)
    data['regime_momentum_score'] = data['high_vol_alignment'] + data['low_vol_alignment']
    
    # Dynamic Factor Integration
    # Core Momentum Components
    data['price_momentum'] = data['net_intraday_bias'] * data['momentum_acceleration']
    data['volume_momentum'] = data['price_volume_alignment'] * data['volume_intensity']
    data['efficiency_momentum_adj'] = data['efficiency_momentum'] * data['daily_efficiency']
    data['flow_momentum'] = data['flow_efficiency'] * data['flow_asymmetry']
    
    # Regime-Adaptive Weighting
    data['volatility_weight'] = data['vol_regime']
    data['compression_weight'] = data['compression_phase'].astype(int) * data['compression_duration']
    data['persistence_weight'] = data['regime_stability']
    data['divergence_weight'] = abs(data['short_vs_medium_divergence'])
    
    # Confirmation Layers
    data['volume_confirmation'] = data['volume_momentum'] * data['volume_intensity']
    data['flow_confirmation'] = data['flow_momentum'] * data['flow_asymmetry']
    data['efficiency_confirmation'] = data['efficiency_momentum_adj'] * data['range_compression']
    data['overall_confirmation'] = data['volume_confirmation'] * data['flow_confirmation'] * data['efficiency_confirmation']
    
    # Composite Alpha Construction
    # Raw Factor Combination
    data['regime_weighted_momentum'] = data['price_momentum'] * data['volatility_weight']
    data['volume_confirmed_momentum'] = data['volume_momentum'] * data['volume_confirmation']
    data['efficiency_enhanced_momentum'] = data['efficiency_momentum_adj'] * data['efficiency_confirmation']
    data['flow_integrated_momentum'] = data['flow_momentum'] * data['flow_confirmation']
    
    # Divergence Enhancement
    data['cross_timeframe_boost'] = (data['regime_weighted_momentum'] + 
                                   data['volume_confirmed_momentum'] + 
                                   data['efficiency_enhanced_momentum'] + 
                                   data['flow_integrated_momentum']) * data['divergence_weight']
    
    data['compression_breakout_boost'] = data['cross_timeframe_boost'] * data['compression_weight']
    data['persistence_filtered'] = data['compression_breakout_boost'] * data['persistence_weight']
    
    # Risk and Quality Adjustment
    # Filter high volatility without clear direction
    high_vol_no_direction = (data['regime_class'] == 'high') & (abs(data['net_intraday_bias']) < 0.1)
    # Filter low-quality volume signals
    low_volume_quality = data['volume_intensity'] < 0.5
    # Filter short-lived momentum patterns
    short_momentum = abs(data['immediate_momentum']) > 0.05
    # Filter inconsistent flow patterns
    inconsistent_flow = abs(data['flow_asymmetry']) > 10
    
    # Apply filters
    filter_mask = ~(high_vol_no_direction | low_volume_quality | short_momentum | inconsistent_flow)
    data['filtered_alpha'] = data['persistence_filtered'] * filter_mask.astype(int)
    
    # Final Alpha Output
    data['core_alpha'] = (data['regime_weighted_momentum'] * 
                         data['volume_confirmed_momentum'] * 
                         data['efficiency_enhanced_momentum'] * 
                         data['flow_integrated_momentum'])
    
    data['confirmed_alpha'] = data['core_alpha'] * data['overall_confirmation']
    
    # Return the final alpha factor
    alpha = data['confirmed_alpha'].fillna(0)
    
    return alpha
