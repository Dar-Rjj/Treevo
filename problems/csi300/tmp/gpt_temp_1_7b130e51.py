import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Volume-Price Efficiency
    # Short-Term (3-day) Efficiency Ratio
    data['vwap_3d'] = (data['volume'].rolling(window=3).apply(
        lambda x: (x * data.loc[x.index, 'close']).sum() / x.sum(), raw=False
    ))
    data['vol_weighted_change_3d'] = data['vwap_3d'] / data['close'].shift(3) - 1
    data['raw_change_3d'] = data['close'] / data['close'].shift(3) - 1
    data['efficiency_ratio_3d'] = data['vol_weighted_change_3d'] / np.where(data['raw_change_3d'] != 0, data['raw_change_3d'], np.nan)
    
    # Medium-Term (10-day) Efficiency Ratio
    data['vwap_10d'] = (data['volume'].rolling(window=10).apply(
        lambda x: (x * data.loc[x.index, 'close']).sum() / x.sum(), raw=False
    ))
    data['vol_weighted_change_10d'] = data['vwap_10d'] / data['close'].shift(10) - 1
    data['raw_change_10d'] = data['close'] / data['close'].shift(10) - 1
    data['efficiency_ratio_10d'] = data['vol_weighted_change_10d'] / np.where(data['raw_change_10d'] != 0, data['raw_change_10d'], np.nan)
    
    # Volume-Price Acceleration
    data['efficiency_change'] = data['efficiency_ratio_3d'] - data['efficiency_ratio_10d']
    data['acceleration_signal'] = data['efficiency_change'] * data['raw_change_3d']
    
    # Regime Detection via Intraday Price Behavior
    # Opening Gap Analysis
    data['gap_size'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_filled_ratio'] = abs(data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1))
    data['gap_efficiency'] = 1 - data['gap_filled_ratio']
    
    # Session Momentum Analysis
    data['intraday_range'] = data['high'] - data['low']
    data['intraday_range_util'] = abs(data['close'] - data['open']) / np.where(data['intraday_range'] != 0, data['intraday_range'], np.nan)
    data['session_continuation'] = np.sign(data['close'] - data['open']) * np.sign(data['open'] - data['close'].shift(1))
    data['momentum_efficiency'] = data['intraday_range_util'] * data['session_continuation']
    
    # Regime Classification & Weighting
    conditions = [
        (data['gap_efficiency'] > data['gap_efficiency'].rolling(window=20).mean()) & (data['momentum_efficiency'] > 0),
        (data['gap_efficiency'] > data['gap_efficiency'].rolling(window=20).mean()) & (data['momentum_efficiency'] < 0),
        (data['gap_efficiency'] <= data['gap_efficiency'].rolling(window=20).mean())
    ]
    choices = [1.3, 1.1, 0.7]
    data['regime_weight'] = np.select(conditions, choices, default=1.0)
    
    # Microstructure Pressure Confirmation
    # Amount-Based Pressure
    data['daily_pressure'] = data['amount'] * (data['close'] - (data['high'] + data['low']) / 2)
    data['cumulative_pressure_3d'] = data['daily_pressure'].rolling(window=3).sum()
    data['total_amount_3d'] = data['amount'].rolling(window=3).sum()
    data['normalized_pressure'] = data['cumulative_pressure_3d'] / np.where(data['total_amount_3d'] != 0, data['total_amount_3d'], np.nan)
    
    # Volume-Position Pressure
    data['daily_position'] = ((data['close'] - data['low']) / np.where(data['intraday_range'] != 0, data['intraday_range'], np.nan)) * data['volume']
    data['avg_position_3d'] = data['daily_position'].rolling(window=3).mean()
    data['avg_volume_3d'] = data['volume'].rolling(window=3).mean()
    data['normalized_position'] = data['avg_position_3d'] / np.where(data['avg_volume_3d'] != 0, data['avg_volume_3d'], np.nan)
    
    # Microstructure Signal
    microstructure_conditions = [
        (data['normalized_pressure'] > 0) & (data['normalized_position'] > 0.6),
        (data['normalized_pressure'] < 0) & (data['normalized_position'] < 0.4)
    ]
    microstructure_choices = [1.2, 0.8]
    data['microstructure_signal'] = np.select(microstructure_conditions, microstructure_choices, default=1.0)
    
    # Range Position Confirmation
    data['range_position'] = (data['close'] - data['low']) / np.where(data['intraday_range'] != 0, data['intraday_range'], np.nan)
    
    # Adaptive Alpha Synthesis
    data['base_efficiency_factor'] = data['acceleration_signal'] * data['regime_weight']
    data['final_alpha'] = data['base_efficiency_factor'] * data['microstructure_signal'] * data['range_position']
    
    return data['final_alpha']
