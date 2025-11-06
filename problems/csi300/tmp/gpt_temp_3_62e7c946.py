import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate Fractal Momentum Efficiency
    # 3-day efficiency ratio
    data['abs_change_3d'] = abs(data['close'] - data['close'].shift(3))
    data['cum_movement_3d'] = (abs(data['close'] - data['close'].shift(1)) + 
                              abs(data['close'].shift(1) - data['close'].shift(2)) + 
                              abs(data['close'].shift(2) - data['close'].shift(3)))
    data['efficiency_3d'] = data['abs_change_3d'] / data['cum_movement_3d']
    
    # 5-day efficiency ratio
    data['abs_change_5d'] = abs(data['close'] - data['close'].shift(5))
    data['cum_movement_5d'] = (abs(data['close'] - data['close'].shift(1)) + 
                              abs(data['close'].shift(1) - data['close'].shift(2)) + 
                              abs(data['close'].shift(2) - data['close'].shift(3)) + 
                              abs(data['close'].shift(3) - data['close'].shift(4)) + 
                              abs(data['close'].shift(4) - data['close'].shift(5)))
    data['efficiency_5d'] = data['abs_change_5d'] / data['cum_movement_5d']
    
    # Efficiency gradient and momentum direction
    data['efficiency_diff'] = data['efficiency_5d'] - data['efficiency_3d']
    data['momentum_dir'] = np.sign(data['close'] - data['close'].shift(1))
    
    # Volatility environment
    data['recent_vol'] = data['close'].rolling(window=5).std()
    data['baseline_vol'] = data['close'].rolling(window=20).std()
    data['vol_regime'] = data['recent_vol'] / data['baseline_vol']
    
    # Volatility-adjusted returns
    data['vol_adj_return'] = (data['close'] - data['close'].shift(1)) / data['recent_vol']
    data['positive_vol_adj_count'] = data['vol_adj_return'].rolling(window=5).apply(
        lambda x: (x > 0).sum(), raw=True
    )
    
    # Combine efficiency with volatility adaptation
    data['fractal_momentum'] = (data['efficiency_diff'] * data['momentum_dir'] * 
                               data['vol_regime'] * data['positive_vol_adj_count'])
    
    # Calculate Pressure-Turnover Alignment
    # Intraday pressure components
    data['opening_pressure'] = (data['open'] - data['close'].shift(1)) * data['volume']
    data['closing_pressure'] = (data['close'] - data['open']) * data['volume']
    data['net_pressure'] = data['closing_pressure'] - data['opening_pressure']
    
    # Multi-scale turnover dynamics
    data['daily_turnover'] = data['volume'] * data['close']
    data['turnover_3d_avg'] = data['daily_turnover'].rolling(window=3).mean()
    data['turnover_8d_avg'] = data['daily_turnover'].rolling(window=8).mean()
    data['turnover_momentum'] = (data['turnover_3d_avg'] / data['turnover_8d_avg']) - 1
    
    # Pressure ratio and consistency
    data['pressure_ratio'] = data['closing_pressure'] / data['opening_pressure']
    data['pressure_consistency'] = data['net_pressure'].rolling(window=5).apply(
        lambda x: (np.sign(x) == np.sign(x.iloc[-1])).sum() if len(x) == 5 else 0, raw=False
    )
    
    # Pressure-turnover signal
    data['pressure_turnover'] = (data['net_pressure'] * data['pressure_ratio'] * 
                                data['turnover_momentum'] * data['pressure_consistency'])
    
    # Calculate Volume-Weighted Fractal Asymmetry
    # Pressure asymmetry components
    data['upside_pressure'] = (data['high'] - data['close']) * data['volume']
    data['downside_pressure'] = (data['close'] - data['low']) * data['volume']
    
    data['upside_pressure_5d'] = data['upside_pressure'].rolling(window=5).mean()
    data['downside_pressure_5d'] = data['downside_pressure'].rolling(window=5).mean()
    data['pressure_asymmetry'] = (np.log1p(data['upside_pressure_5d']) - 
                                 np.log1p(data['downside_pressure_5d']))
    
    # Volume-price efficiency
    data['price_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['volume_5d_sum'] = data['volume'].rolling(window=5).sum()
    data['volume_concentration'] = data['volume'] / data['volume_5d_sum']
    data['efficiency_volume'] = data['price_efficiency'] * data['volume_concentration']
    
    # Volume breakout and consistency
    data['volume_10d_avg'] = data['volume'].rolling(window=10).mean()
    data['volume_breakout'] = data['volume'] > (1.2 * data['volume_10d_avg'])
    
    data['volume_pct_change'] = data['volume'].pct_change()
    data['volume_consistency'] = data['volume_pct_change'].rolling(window=6).std()
    
    # Asymmetry signal
    data['asymmetry_signal'] = data['pressure_asymmetry'] * data['efficiency_volume']
    data['asymmetry_signal'] = np.where(data['volume_breakout'], 
                                      data['asymmetry_signal'] * 1.8, 
                                      data['asymmetry_signal'])
    data['asymmetry_signal'] = data['asymmetry_signal'] * data['volume_consistency']
    
    # Detect Range Expansion with Volume Confirmation
    # Range dynamics
    data['current_range'] = data['high'] - data['low']
    data['historical_range'] = (data['high'] - data['low']).rolling(window=10).mean()
    data['range_expansion'] = data['current_range'] / data['historical_range']
    
    # Volume cluster enhancement
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_cluster'] = data['volume'] > (2 * data['volume_20d_avg'])
    data['volume_confirmation'] = data['volume'] / data['volume'].shift(1)
    data['cluster_intensity'] = data['volume'] / data['volume_20d_avg']
    
    # Momentum acceleration
    data['momentum_acceleration'] = ((data['close'] / data['close'].shift(1)) / 
                                   (data['close'].shift(1) / data['close'].shift(2)))
    
    # Expansion signal
    data['expansion_signal'] = (data['range_expansion'] * data['momentum_acceleration'] * 
                               data['volume_confirmation'])
    data['expansion_signal'] = np.where(data['volume_cluster'], 
                                      data['expansion_signal'] * data['cluster_intensity'], 
                                      data['expansion_signal'])
    
    # Combine All Components with Adaptive Weighting
    # Volatility regime classification
    data['volatility_class'] = pd.cut(data['vol_regime'], 
                                    bins=[0, 0.8, 1.2, float('inf')], 
                                    labels=['low', 'medium', 'high'])
    
    # Component weights based on volatility regime
    conditions = [
        data['volatility_class'] == 'high',
        data['volatility_class'] == 'low',
        data['volatility_class'] == 'medium'
    ]
    
    fractal_weights = [0.4, 0.2, 0.3]
    pressure_weights = [0.3, 0.2, 0.25]
    asymmetry_weights = [0.1, 0.4, 0.25]
    expansion_weights = [0.2, 0.2, 0.2]
    
    data['fractal_weight'] = np.select(conditions, fractal_weights, default=0.3)
    data['pressure_weight'] = np.select(conditions, pressure_weights, default=0.25)
    data['asymmetry_weight'] = np.select(conditions, asymmetry_weights, default=0.25)
    data['expansion_weight'] = np.select(conditions, expansion_weights, default=0.2)
    
    # Weighted combination
    data['combined_signal'] = (
        data['fractal_momentum'] * data['fractal_weight'] +
        data['pressure_turnover'] * data['pressure_weight'] +
        data['asymmetry_signal'] * data['asymmetry_weight'] +
        data['expansion_signal'] * data['expansion_weight']
    )
    
    # Volume cluster amplification
    data['final_signal'] = np.where(data['volume_cluster'], 
                                  data['combined_signal'] * 1.8, 
                                  data['combined_signal'])
    
    # Volatility-based smoothing
    conditions_smooth = [
        data['volatility_class'] == 'high',
        data['volatility_class'] == 'low',
        data['volatility_class'] == 'medium'
    ]
    
    smooth_windows = [3, 8, 5]
    
    data['alpha_factor'] = np.nan
    for condition, window in zip(conditions_smooth, smooth_windows):
        mask = condition
        data.loc[mask, 'alpha_factor'] = data.loc[mask, 'final_signal'].rolling(window=window).mean()
    
    # Fill any remaining NaN values with 5-day moving average
    data['alpha_factor'] = data['alpha_factor'].fillna(data['final_signal'].rolling(window=5).mean())
    
    return data['alpha_factor']
