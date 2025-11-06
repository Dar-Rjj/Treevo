import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Asymmetric Momentum Acceleration with Volatility-Microstructure Divergence
    """
    data = df.copy()
    
    # Asymmetric Momentum Acceleration Component
    # Multi-scale momentum asymmetry
    data['short_up_momentum'] = np.where(data['close'] > data['close'].shift(5), 
                                        data['close'] - data['close'].shift(5), 0)
    data['short_down_momentum'] = np.where(data['close'] < data['close'].shift(5), 
                                          data['close'].shift(5) - data['close'], 0)
    data['medium_up_momentum'] = np.where(data['close'] > data['close'].shift(20), 
                                         data['close'] - data['close'].shift(20), 0)
    data['medium_down_momentum'] = np.where(data['close'] < data['close'].shift(20), 
                                           data['close'].shift(20) - data['close'], 0)
    
    # Avoid division by zero
    short_up_ratio = data['short_up_momentum'] / (data['short_down_momentum'] + 1e-8)
    medium_up_ratio = data['medium_up_momentum'] / (data['medium_down_momentum'] + 1e-8)
    data['momentum_acceleration_asymmetry'] = short_up_ratio - medium_up_ratio
    
    # Volume-confirmed momentum acceleration
    data['volume_acceleration'] = data['volume'] / (data['volume'].shift(5) + 1e-8) - 1
    
    # Volume asymmetry
    up_day_mask = data['close'] > data['close'].shift(1)
    down_day_mask = data['close'] < data['close'].shift(1)
    
    up_volume_growth = data.loc[up_day_mask, 'volume'] / (data.loc[up_day_mask, 'volume'].shift(5) + 1e-8) - 1
    down_volume_growth = data.loc[down_day_mask, 'volume'] / (data.loc[down_day_mask, 'volume'].shift(5) + 1e-8) - 1
    
    data['volume_asymmetry'] = 0
    data.loc[up_day_mask, 'volume_asymmetry'] = up_volume_growth / (abs(down_volume_growth) + 1e-8)
    data.loc[down_day_mask, 'volume_asymmetry'] = abs(down_volume_growth) / (up_volume_growth + 1e-8)
    
    # Volatility-Efficient Microstructure Divergence
    # Asymmetric volatility efficiency
    data['upward_efficiency'] = np.where(data['close'] > data['close'].shift(1),
                                        (data['close'] - data['close'].shift(1)) / 
                                        (data['high'] - data['close'].shift(1) + 1e-8), 0)
    data['downward_efficiency'] = np.where(data['close'] < data['close'].shift(1),
                                          (data['close'].shift(1) - data['close']) / 
                                          (data['close'].shift(1) - data['low'] + 1e-8), 0)
    
    data['efficiency_asymmetry'] = data['upward_efficiency'] / (data['downward_efficiency'] + 1e-8)
    
    # Microstructure flow divergence
    data['order_flow'] = data['volume'] * (data['close'] - data['close'].shift(1))
    
    # 3-day cumulative asymmetric flow
    for i in range(len(data)):
        if i >= 3:
            window_data = data.iloc[i-3:i+1]
            up_flow = window_data.loc[window_data['close'] > window_data['close'].shift(1), 'order_flow'].sum()
            down_flow = window_data.loc[window_data['close'] < window_data['close'].shift(1), 'order_flow'].sum()
            data.loc[data.index[i], 'flow_asymmetry'] = up_flow / (abs(down_flow) + 1e-8)
        else:
            data.loc[data.index[i], 'flow_asymmetry'] = 1.0
    
    # Multi-scale Amount Flow Confirmation
    # Asymmetric amount momentum
    up_amount_growth = data.loc[data['close'] > data['close'].shift(5), 'amount'] / \
                      (data.loc[data['close'] > data['close'].shift(5), 'amount'].shift(5) + 1e-8) - 1
    down_amount_growth = data.loc[data['close'] < data['close'].shift(5), 'amount'] / \
                        (data.loc[data['close'] < data['close'].shift(5), 'amount'].shift(5) + 1e-8) - 1
    
    data['amount_asymmetry'] = 0
    data.loc[data['close'] > data['close'].shift(5), 'amount_asymmetry'] = \
        up_amount_growth / (abs(down_amount_growth) + 1e-8)
    data.loc[data['close'] < data['close'].shift(5), 'amount_asymmetry'] = \
        abs(down_amount_growth) / (up_amount_growth + 1e-8)
    
    # Dynamic Breakout Sustainability Assessment
    # Asymmetric true range
    data['asymmetric_true_range'] = (data['high'] - data['close'].shift(1)) / \
                                   (data['close'].shift(1) - data['low'] + 1e-8)
    
    # Volatility expansion using 20-day median true range
    data['true_range'] = data['high'] - data['low']
    data['median_true_range'] = data['true_range'].rolling(window=20, min_periods=1).median()
    data['volatility_expansion'] = data['true_range'] / (data['median_true_range'] + 1e-8)
    
    # Multi-scale momentum alignment
    data['momentum_alignment'] = np.sign(data['momentum_acceleration_asymmetry']) * \
                                np.sign(data['volume_asymmetry'] - 1) * \
                                np.sign(data['amount_asymmetry'] - 1)
    
    # Microstructure flow intensity
    data['flow_intensity'] = data['order_flow'].abs().rolling(window=5, min_periods=1).mean()
    
    # Breakout quality score
    data['breakout_quality'] = (data['volatility_expansion'] * 
                               data['momentum_alignment'] * 
                               (data['flow_intensity'] / (data['flow_intensity'].rolling(window=20, min_periods=1).mean() + 1e-8)))
    
    # Composite Factor Synthesis
    # Dynamic asymmetric divergence scoring
    divergence_score = (data['momentum_acceleration_asymmetry'] * 
                       data['efficiency_asymmetry'] * 
                       data['flow_asymmetry'])
    
    # Weight by microstructure flow divergence
    flow_divergence_weight = np.where(data['flow_asymmetry'] > 1, 
                                     data['flow_asymmetry'], 
                                     1 / (data['flow_asymmetry'] + 1e-8))
    
    # Adjust for multi-scale amount flow confirmation
    amount_confirmation = np.where(data['amount_asymmetry'] > 1, 1, -1)
    
    # Final composite factor
    composite_factor = (divergence_score * 
                       flow_divergence_weight * 
                       amount_confirmation * 
                       data['breakout_quality'])
    
    # Normalize and return
    factor = (composite_factor - composite_factor.rolling(window=20, min_periods=1).mean()) / \
             (composite_factor.rolling(window=20, min_periods=1).std() + 1e-8)
    
    return factor
