import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Fractal Efficiency Calculation
    # Ultra-Short Fractal Efficiency (3-day)
    data['range_sum_3'] = data['high'].rolling(window=3).apply(lambda x: (x - data.loc[x.index, 'low']).sum(), raw=False)
    data['close_path_3'] = data['close'].diff().abs().rolling(window=2).sum()
    data['fractal_dim_3'] = np.log(data['range_sum_3']) / np.log(data['close_path_3'].replace(0, np.nan))
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['ultra_short_fe'] = data['fractal_dim_3'] * data['intraday_efficiency']
    
    # Short-Term Fractal Efficiency (8-day)
    data['range_sum_8'] = data['high'].rolling(window=8).apply(lambda x: (x - data.loc[x.index, 'low']).sum(), raw=False)
    data['close_path_8'] = data['close'].diff().abs().rolling(window=8).sum()
    data['fractal_dim_8'] = np.log(data['range_sum_8']) / np.log(data['close_path_8'].replace(0, np.nan))
    data['price_trend_5'] = data['close'] / data['close'].shift(5) - 1
    data['short_term_fe'] = data['fractal_dim_8'] * data['price_trend_5']
    
    # Fractal Efficiency Acceleration Signal
    conditions = [
        (data['ultra_short_fe'] > data['short_term_fe']) & (data['ultra_short_fe'] > 0),
        (data['ultra_short_fe'] < data['short_term_fe']) & (data['ultra_short_fe'] < 0)
    ]
    choices = [1, -1]
    data['fractal_acceleration'] = np.select(conditions, choices, default=0)
    
    # Volume Compression Divergence Detection
    # Volatility Entropy Calculation
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # 8-day True Range percentile rank
    data['tr_percentile'] = data['true_range'].rolling(window=20).rank(pct=True)
    
    # Shannon entropy calculation
    def calculate_entropy(series):
        value_counts = series.value_counts(normalize=True, bins=10)
        return -np.sum(value_counts * np.log(value_counts.replace(0, np.nan)))
    
    data['volatility_entropy'] = data['tr_percentile'].rolling(window=8).apply(calculate_entropy, raw=False)
    
    # Volume Compression Analysis
    data['volume_std_5'] = data['volume'].rolling(window=5).std()
    data['volume_mean_5'] = data['volume'].rolling(window=5).mean()
    data['volume_compression_index'] = data['volume_std_5'] / data['volume_mean_5'].replace(0, np.nan)
    
    data['volume_momentum'] = data['volume'] / data['volume'].rolling(window=4).mean().shift(1).replace(0, np.nan)
    data['volume_to_range'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Compression Divergence Signal
    data['compression_divergence'] = data['volatility_entropy'] - data['volume_compression_index']
    data['volume_momentum_div'] = data['volume_momentum'] - data['volume_compression_index']
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Adaptive Fractal Momentum System
    # 3-day Fractal Momentum
    data['fractal_momentum_3'] = data['fractal_dim_3'] * data['intraday_efficiency'] * data['volume_momentum']
    
    # Efficiency-Weighted Breakout Detection
    data['max_high_5'] = data['high'].rolling(window=5).max()
    data['min_low_5'] = data['low'].rolling(window=5).min()
    data['movement_efficiency'] = (data['close'] - data['close'].shift(5)) / (data['max_high_5'] - data['min_low_5']).replace(0, np.nan)
    
    data['short_breakout'] = data['close'] / data['high'].rolling(window=5).max()
    data['medium_breakout'] = data['close'] / data['high'].rolling(window=10).max()
    data['long_breakout'] = data['close'] / data['high'].rolling(window=20).max()
    
    data['efficiency_weighted_breakout'] = (data['movement_efficiency'] * 
                                          (data['short_breakout'] + data['medium_breakout'] + data['long_breakout'] - 3)) / 3
    
    # Fractal Momentum Divergence
    data['fractal_momentum_div'] = (data['ultra_short_fe'] - data['short_term_fe']) * data['volume_compression_index']
    
    # Adaptive Momentum Integration
    data['adaptive_momentum'] = (data['fractal_momentum_3'] * data['efficiency_weighted_breakout']) / abs(data['fractal_momentum_div']).replace(0, np.nan)
    data['adaptive_momentum'] = data['adaptive_momentum'] * data['volume_to_range']
    data['adaptive_momentum'] = np.tanh(data['adaptive_momentum'])
    
    # Volume-Price Regime Detection
    conditions_regime = [
        (data['volume_compression_index'] < 0.5) & (data['efficiency_weighted_breakout'] > 0) & (data['volume_momentum'] > 1.2),
        (data['volume_compression_index'] > 1.5) & (data['efficiency_weighted_breakout'] < 0) & (data['volume_momentum'] < 0.8)
    ]
    choices_regime = ['high_compression', 'low_compression']
    data['regime'] = np.select(conditions_regime, choices_regime, default='transition')
    
    # Component Calculations
    # Fractal Efficiency Component
    data['fractal_efficiency_component'] = data['fractal_acceleration'] * data['compression_divergence'] * data['volume_momentum']
    
    # Fractal-Momentum Component
    data['fractal_momentum_component'] = data['adaptive_momentum'] * data['volume_to_range'] * data['volatility_entropy']
    
    # Breakout Component
    data['breakout_component'] = data['efficiency_weighted_breakout'] * data['volume_momentum'] * data['fractal_dim_8']
    
    # Regime-Based Weighting
    conditions_weights_fe = [
        data['regime'] == 'high_compression',
        data['regime'] == 'low_compression',
        data['regime'] == 'transition'
    ]
    weights_fe = [0.4, 0.6, 0.5]
    data['weight_fe'] = np.select(conditions_weights_fe, weights_fe)
    
    conditions_weights_fm = [
        data['regime'] == 'high_compression',
        data['regime'] == 'low_compression',
        data['regime'] == 'transition'
    ]
    weights_fm = [0.7, 0.4, 0.6]
    data['weight_fm'] = np.select(conditions_weights_fm, weights_fm)
    
    conditions_weights_bo = [
        data['regime'] == 'high_compression',
        data['regime'] == 'low_compression',
        data['regime'] == 'transition'
    ]
    weights_bo = [0.9, 0.3, 0.6]
    data['weight_bo'] = np.select(conditions_weights_bo, weights_bo)
    
    # Weighted Components
    data['weighted_fe'] = data['fractal_efficiency_component'] * data['weight_fe']
    data['weighted_fm'] = data['fractal_momentum_component'] * data['weight_fm']
    data['weighted_bo'] = data['breakout_component'] * data['weight_bo']
    
    # Fractal Divergence Enhancement
    conditions_divergence = [
        (data['fractal_acceleration'] > 0) & (data['volume_compression_index'] < 0.5),
        (data['fractal_acceleration'] < 0) & (data['volume_compression_index'] > 1.5)
    ]
    divergence_multipliers = [1.5, 0.5]
    data['divergence_multiplier'] = np.select(conditions_divergence, divergence_multipliers, default=1.0)
    
    # Final Alpha Factor
    data['composite_factor'] = (data['weighted_fe'] + data['weighted_fm'] + data['weighted_bo']) * data['volume_to_range']
    data['final_alpha'] = data['composite_factor'] * data['divergence_multiplier']
    
    return data['final_alpha']
