import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Gap-Momentum Acceleration Dynamics
    # Gap Breakout Components
    data['gap_acceleration'] = (data['close']/data['open'] - 1) - (data['close'].shift(3)/data['open'].shift(3) - 1)
    data['range_adjusted_breakout'] = (data['high'] - data['close'].shift(1) - (data['close'].shift(1) - data['low'])) / (data['high'] - data['low'])
    data['gap_direction_persistence'] = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['close'] - data['open'])
    
    # Momentum Acceleration Integration
    data['close_momentum_acc'] = (data['close']/data['close'].shift(1)) / (data['close'].shift(1)/data['close'].shift(2))
    data['high_momentum_acc'] = (data['high']/data['high'].shift(1)) / (data['high'].shift(1)/data['high'].shift(2))
    data['low_momentum_acc'] = (data['low']/data['low'].shift(1)) / (data['low'].shift(1)/data['low'].shift(2))
    
    # Gap-Momentum Alignment
    data['primary_alignment'] = data['gap_acceleration'] * ((data['close']/data['close'].shift(5) - 1) - (data['close']/data['close'].shift(10) - 1))
    
    # Directional Divergence Analysis
    data['bullish_divergence'] = ((data['high_momentum_acc'] > data['close_momentum_acc']) & 
                                 (data['close_momentum_acc'] > data['low_momentum_acc'])).astype(float)
    data['bearish_divergence'] = ((data['low_momentum_acc'] > data['close_momentum_acc']) & 
                                 (data['close_momentum_acc'] > data['high_momentum_acc'])).astype(float)
    data['convergence_signal'] = (abs(data['high_momentum_acc'] - data['low_momentum_acc']) < 0.01).astype(float)
    
    # Multi-Dimensional Volume Confirmation
    # Volume Momentum Analysis
    data['raw_volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_acceleration'] = (data['volume']/data['volume'].shift(1)) / (data['volume'].shift(1)/data['volume'].shift(2))
    
    # Volume Persistence
    volume_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        volume_persistence.iloc[i] = ((data['volume'].iloc[i-4:i+1] > data['volume'].shift(1).iloc[i-4:i+1]).sum())
    data['volume_persistence'] = volume_persistence
    
    # Volume-Weighted Efficiency
    data['base_efficiency'] = (data['volume']/data['volume'].shift(1)) * abs((data['close'] - data['low'])/(data['high'] - data['low']) - 0.5)
    
    # Price-Volume Efficiency Components
    up_mask = data['close'] > data['open']
    down_mask = data['close'] < data['open']
    data['up_day_efficiency'] = np.where(up_mask, (data['close'] - data['open']) / data['amount'], 0)
    data['down_day_efficiency'] = np.where(down_mask, (data['open'] - data['close']) / data['amount'], 0)
    data['efficiency_ratio'] = np.where(data['down_day_efficiency'] != 0, 
                                       data['up_day_efficiency'] / data['down_day_efficiency'], 1)
    
    # Range Expansion Breakout Analysis
    # Intraday Range Dynamics
    data['high_low_range'] = data['high'] - data['low']
    data['high_prev_close'] = abs(data['high'] - data['close'].shift(1))
    data['low_prev_close'] = abs(data['low'] - data['close'].shift(1))
    
    data['range_momentum'] = data['high_low_range'] / data['high_low_range'].shift(1)
    data['range_acceleration'] = data['range_momentum'] / data['range_momentum'].shift(1)
    data['range_volatility'] = data[['high_low_range', 'high_prev_close', 'low_prev_close']].max(axis=1)
    
    # Range Breakout Signals
    data['upper_breakout'] = ((data['high'] > data['high'].shift(1)) & (data['range_acceleration'] > 1.2)).astype(float)
    data['lower_breakout'] = ((data['low'] < data['low'].shift(1)) & (data['range_acceleration'] > 1.2)).astype(float)
    data['range_compression'] = ((data['range_momentum'] < 0.8) & (data['range_acceleration'] < 1.0)).astype(float)
    
    # Gap-Range Integration
    data['opening_gap_magnitude'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_filling_efficiency'] = abs(data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1))
    
    # Range-Volume Confirmation
    avg_range = data['high_low_range'].rolling(window=20, min_periods=10).mean()
    avg_volume = data['volume'].rolling(window=20, min_periods=10).mean()
    
    data['high_range_high_volume'] = ((data['high_low_range'] > 1.2 * avg_range) & 
                                     (data['volume'] > 1.2 * avg_volume)).astype(float)
    data['low_range_low_volume'] = ((data['high_low_range'] < 0.8 * avg_range) & 
                                   (data['volume'] < 0.8 * avg_volume)).astype(float)
    data['range_volume_divergence'] = data['range_momentum'] / (data['raw_volume_momentum'] + 1e-8)
    
    # Multi-Timeframe Momentum Convergence
    # Short-term Momentum (1-3 days)
    data['momentum_1d'] = data['close'] / data['close'].shift(1) - 1
    data['momentum_2d'] = data['close'] / data['close'].shift(2) - 1
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['short_term_acceleration'] = data['momentum_3d'] / (data['momentum_1d'] + 1e-8)
    
    # Medium-term Momentum (5-10 days)
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['medium_term_acceleration'] = data['momentum_10d'] / (data['momentum_5d'] + 1e-8)
    
    # Momentum Convergence Framework
    data['short_medium_convergence'] = (np.sign(data['short_term_acceleration']) == 
                                       np.sign(data['medium_term_acceleration'])).astype(float)
    data['momentum_divergence'] = (abs(data['short_term_acceleration'] - data['medium_term_acceleration']) > 0.05).astype(float)
    data['timeframe_alignment'] = ((np.sign(data['momentum_3d']) == np.sign(data['momentum_10d']))).astype(float)
    
    # Adaptive Signal Synthesis
    # Signal Quality Assessment
    data['gap_momentum_quality'] = (
        abs(data['gap_acceleration']) + 
        abs(data['range_adjusted_breakout']) + 
        abs(data['gap_direction_persistence'])
    )
    
    data['volume_confirmation_quality'] = (
        data['base_efficiency'] + 
        data['volume_persistence'] + 
        abs(data['efficiency_ratio'])
    )
    
    data['range_breakout_quality'] = (
        data['range_momentum'] + 
        (data['upper_breakout'] + data['lower_breakout']) + 
        (data['high_range_high_volume'] + data['low_range_low_volume'])
    )
    
    # Multi-Dimensional Weighting
    primary_weight = abs(data['primary_alignment'])
    volume_weight = data['volume_confirmation_quality']
    range_weight = data['range_breakout_quality']
    timeframe_weight = data['short_medium_convergence'] + data['timeframe_alignment']
    
    # Risk Adjustment Factors
    range_volatility_adj = 1 / (data['range_volatility'] + 1e-8)
    
    # Volume Spike Clustering
    volume_spike_clustering = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        volume_spike_clustering.iloc[i] = ((data['volume'].iloc[i-4:i+1] > 1.8 * data['volume'].shift(1).iloc[i-4:i+1]).sum())
    volume_spike_adj = 1 / (1 + volume_spike_clustering)
    
    gap_risk_adj = 1 / (1 + abs(data['opening_gap_magnitude']))
    
    # Final Alpha Generation
    base_signal = (
        data['primary_alignment'] * primary_weight +
        data['bullish_divergence'] - data['bearish_divergence'] +
        data['convergence_signal'] * 0.5
    )
    
    confirmation_multiplier = volume_weight * range_weight * timeframe_weight
    risk_adjustment = range_volatility_adj * volume_spike_adj * gap_risk_adj
    
    final_alpha = base_signal * confirmation_multiplier * risk_adjustment
    
    return final_alpha
