import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility Regime Classification
    # Daily Range
    data['daily_range'] = data['high'] - data['low']
    
    # 5-day Average Range
    data['range_5d_avg'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    
    # 20-day Median Range
    data['range_20d_median'] = data['daily_range'].rolling(window=20, min_periods=10).median()
    
    # Regime Classification
    data['vol_regime'] = 'normal'
    high_vol_condition = data['range_5d_avg'] > (1.3 * data['range_20d_median'])
    low_vol_condition = data['range_5d_avg'] < (0.8 * data['range_20d_median'])
    data.loc[high_vol_condition, 'vol_regime'] = 'high'
    data.loc[low_vol_condition, 'vol_regime'] = 'low'
    
    # 2. Microstructure Anchor Construction
    # Volume-Weighted Anchor Points
    data['daily_anchor'] = (data['high'] * data['volume'] + data['low'] * data['volume']) / (2 * data['volume'])
    data['anchor_5d_ma'] = data['daily_anchor'].rolling(window=5, min_periods=3).mean()
    data['anchor_deviation'] = (data['close'] - data['daily_anchor']) / data['close']
    
    # Gap-Based Anchors
    data['opening_gap'] = (data['open'] / data['close'].shift(1)) - 1
    data['intraday_pressure'] = (data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'])
    
    # Gap Persistence
    data['gap_sign'] = np.sign(data['opening_gap'])
    data['gap_persistence'] = 0
    for i in range(1, len(data)):
        if data['gap_sign'].iloc[i] == data['gap_sign'].iloc[i-1] and data['gap_sign'].iloc[i] != 0:
            data['gap_persistence'].iloc[i] = data['gap_persistence'].iloc[i-1] + 1
    
    # 3. Multi-Timeframe Momentum Acceleration
    # Efficiency-Based Momentum
    data['close_3d_change'] = abs(data['close'] - data['close'].shift(3))
    data['range_3d'] = data['high'].rolling(window=3, min_periods=2).max() - data['low'].rolling(window=3, min_periods=2).min()
    data['short_efficiency'] = data['close_3d_change'] / data['range_3d']
    
    data['close_10d_change'] = abs(data['close'] - data['close'].shift(10))
    data['range_10d'] = data['high'].rolling(window=10, min_periods=7).max() - data['low'].rolling(window=10, min_periods=7).min()
    data['medium_efficiency'] = data['close_10d_change'] / data['range_10d']
    
    data['efficiency_acceleration'] = data['short_efficiency'] / data['medium_efficiency']
    
    # Price Acceleration
    data['short_price_acc'] = (data['close']/data['close'].shift(1) - 1) - (data['close'].shift(1)/data['close'].shift(2) - 1)
    data['medium_price_acc'] = (data['close']/data['close'].shift(5) - 1) - (data['close'].shift(5)/data['close'].shift(10) - 1)
    data['acceleration_ratio'] = data['short_price_acc'] / data['medium_price_acc']
    
    # Regime-Adaptive Combination
    data['regime_adaptive_comb'] = 0
    high_vol_mask = data['vol_regime'] == 'high'
    low_vol_mask = data['vol_regime'] == 'low'
    normal_vol_mask = data['vol_regime'] == 'normal'
    
    data.loc[high_vol_mask, 'regime_adaptive_comb'] = data['efficiency_acceleration'] * data['short_price_acc']
    data.loc[low_vol_mask, 'regime_adaptive_comb'] = data['medium_efficiency'] * data['medium_price_acc']
    data.loc[normal_vol_mask, 'regime_adaptive_comb'] = (data['efficiency_acceleration'] + data['acceleration_ratio']) / 2
    
    # 4. Volume Pressure Asymmetry Analysis
    # Directional Volume Concentration
    data['up_day'] = data['close'] > data['open']
    data['down_day'] = data['close'] < data['open']
    
    data['up_volume_3d'] = data.apply(lambda x: x['volume'] if x['up_day'] else 0, axis=1).rolling(window=3, min_periods=2).sum()
    data['down_volume_3d'] = data.apply(lambda x: x['volume'] if x['down_day'] else 0, axis=1).rolling(window=3, min_periods=2).sum()
    
    data['volume_asymmetry'] = np.log(1 + data['up_volume_3d']) - np.log(1 + data['down_volume_3d'])
    
    # Price-Volume Divergence
    data['volume_efficiency_ratio'] = abs(data['close'] - data['close'].shift(3)) / data['volume'].rolling(window=3, min_periods=2).sum()
    data['anchor_divergence'] = data['anchor_deviation'] * data['volume_efficiency_ratio']
    data['divergence_momentum'] = data['anchor_divergence'] - data['anchor_divergence'].shift(3)
    
    # 5. Turnover-Aligned Breakout Detection
    # Turnover Analysis
    data['daily_turnover'] = data['volume'] * data['close']
    data['turnover_5d_avg'] = data['daily_turnover'].rolling(window=5, min_periods=3).mean()
    data['turnover_10d_avg'] = data['daily_turnover'].rolling(window=10, min_periods=7).mean()
    data['turnover_momentum'] = (data['turnover_5d_avg'] / data['turnover_10d_avg']) - 1
    
    # Efficiency Breakout
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['efficiency_3d_avg'] = data['intraday_efficiency'].rolling(window=3, min_periods=2).mean()
    data['breakout_signal'] = (data['intraday_efficiency'] / data['efficiency_3d_avg']) * data['turnover_momentum']
    
    # 6. Microstructure Noise Filtering
    # Price Rejection Analysis
    data['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'])
    data['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'])
    data['net_rejection'] = data['upper_shadow'] - data['lower_shadow']
    
    # Gap Efficiency
    data['overnight_gap'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_gap_closure'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['gap_efficiency_score'] = data['intraday_gap_closure'] / (1 + data['overnight_gap'])
    
    # 7. Adaptive Factor Integration
    # Core Momentum Signal
    data['base_momentum'] = data['regime_adaptive_comb'] * data['anchor_deviation']
    data['volume_enhanced_momentum'] = data['base_momentum'] * (1 + data['volume_asymmetry'])
    
    # Confirmation Signals
    data['divergence_confirmation'] = data['divergence_momentum'] * data['gap_persistence']
    data['breakout_confirmation'] = data['breakout_signal'] * data['turnover_momentum']
    
    # Noise Filter Application
    data['rejection_filter'] = 1 / (1 + abs(data['net_rejection']))
    data['gap_filter'] = data['gap_efficiency_score']
    
    # Regime-Adaptive Weighting
    data['regime_weight'] = 0.5
    data.loc[data['vol_regime'] == 'high', 'regime_weight'] = 0.4
    data.loc[data['vol_regime'] == 'low', 'regime_weight'] = 0.6
    
    # Final Alpha Output
    data['raw_signal'] = data['volume_enhanced_momentum'] * (1 + 0.2 * data['divergence_confirmation']) * (1 + 0.15 * data['breakout_confirmation'])
    data['filtered_signal'] = data['raw_signal'] * data['rejection_filter'] * data['gap_filter']
    data['final_factor'] = data['filtered_signal'] * data['regime_weight']
    
    return data['final_factor']
