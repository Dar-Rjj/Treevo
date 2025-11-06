import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate price accelerations
    # Short-term acceleration: (Close_t - Close_{t-1}) - (Close_{t-1} - Close_{t-2})
    data['price_change_1'] = data['close'].diff(1)
    data['price_change_2'] = data['close'].diff(1).shift(1)
    data['short_accel'] = data['price_change_1'] - data['price_change_2']
    
    # 3-day cumulative acceleration
    data['cumulative_accel_3d'] = data['short_accel'].rolling(window=3, min_periods=1).sum()
    
    # Acceleration magnitude relative to price level
    data['accel_magnitude'] = data['cumulative_accel_3d'] / (data['close'].rolling(window=5, min_periods=1).mean())
    
    # Medium-term acceleration: (Close_t - Close_{t-3}) - (Close_{t-3} - Close_{t-6})
    data['price_change_3'] = data['close'].diff(3)
    data['price_change_6'] = data['close'].diff(3).shift(3)
    data['medium_accel'] = data['price_change_3'] - data['price_change_6']
    
    # 5-day acceleration persistence
    data['accel_persistence'] = data['short_accel'].rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(x > 0) - np.sum(x < 0) if len(x) == 5 else 0
    )
    
    # Volume-weighted calculations
    # Volume-adjusted price movements
    data['intraday_range'] = data['high'] - data['low']
    data['intraday_range'] = data['intraday_range'].replace(0, np.nan)
    data['volume_adjusted_move'] = data['volume'] * (data['close'] - data['open']) / data['intraday_range']
    
    # 5-day volume-weighted momentum
    data['volume_weighted_momentum'] = data['volume_adjusted_move'].rolling(window=5, min_periods=1).sum()
    
    # Volume efficiency ratio
    data['volume_efficiency'] = (data['close'] - data['open']).abs() / data['intraday_range']
    data['volume_efficiency'] = data['volume_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Acceleration-Volume Divergence
    data['accel_direction'] = np.sign(data['cumulative_accel_3d'])
    data['volume_direction'] = np.sign(data['volume_weighted_momentum'])
    
    # Divergence magnitude
    data['direction_divergence'] = data['accel_direction'] * data['volume_direction']
    data['magnitude_divergence'] = (data['cumulative_accel_3d'] - data['volume_weighted_momentum']) / (
        data['cumulative_accel_3d'].abs() + data['volume_weighted_momentum'].abs() + 1e-8
    )
    
    # Divergence persistence
    data['divergence_persistence'] = data['direction_divergence'].rolling(window=3, min_periods=1).sum()
    
    # Generate divergence-based signals
    data['signal_strength'] = 0.0
    
    # Strong acceleration + High volume alignment
    strong_accel_mask = (data['cumulative_accel_3d'].abs() > data['cumulative_accel_3d'].rolling(window=10, min_periods=1).std())
    volume_alignment_mask = (data['direction_divergence'] > 0)
    data.loc[strong_accel_mask & volume_alignment_mask, 'signal_strength'] += 1.0
    
    # Weak acceleration + Volume divergence
    weak_accel_mask = (data['cumulative_accel_3d'].abs() < data['cumulative_accel_3d'].rolling(window=10, min_periods=1).std() * 0.5)
    volume_divergence_mask = (data['direction_divergence'] < 0)
    data.loc[weak_accel_mask & volume_divergence_mask, 'signal_strength'] -= 0.8
    
    # Mixed acceleration patterns - reduce signal
    mixed_mask = (data['accel_persistence'].abs() <= 1)
    data.loc[mixed_mask, 'signal_strength'] *= 0.5
    
    # Signal confidence score
    data['signal_confidence'] = (
        data['volume_efficiency'] * 
        (1 - data['magnitude_divergence'].abs()) * 
        (data['divergence_persistence'].abs() / 3)
    )
    
    # Range efficiency adjustment
    data['range_efficiency'] = (data['close'] - data['open']).abs() / data['intraday_range']
    data['range_efficiency'] = data['range_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Intraday range normalized
    data['intraday_range_norm'] = data['intraday_range'] / ((data['high'] + data['low']) / 2)
    
    # Apply range efficiency to divergence signal
    data['divergence_signal'] = data['signal_strength'] * data['range_efficiency'] * data['signal_confidence']
    
    # Temporal consistency filter
    data['signal_direction'] = np.sign(data['divergence_signal'])
    data['temporal_consistency'] = data['signal_direction'].rolling(window=3, min_periods=1).sum().abs() / 3
    
    # Amount-based scaling
    data['amount_median_15d'] = data['amount'].rolling(window=15, min_periods=1).median()
    data['amount_scaling'] = data['amount'] / data['amount_median_15d']
    data['amount_scaling'] = data['amount_scaling'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Final factor construction
    data['alpha_factor'] = (
        data['divergence_signal'] * 
        data['temporal_consistency'] * 
        data['amount_scaling']
    )
    
    # Clean up intermediate columns
    result = data['alpha_factor'].copy()
    
    return result
