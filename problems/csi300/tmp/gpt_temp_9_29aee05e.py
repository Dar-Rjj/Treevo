import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price Acceleration Calculation
    data['momentum'] = data['close'] / data['close'].shift(1) - 1
    data['acceleration'] = data['momentum'] - data['momentum'].shift(1)
    
    # Multi-period Acceleration
    data['return_5d'] = data['close'] / data['close'].shift(5) - 1
    data['return_10d'] = data['close'] / data['close'].shift(10) - 1
    data['multi_period_accel'] = (data['return_5d'] - data['return_10d']) / 5
    
    # Acceleration Persistence Analysis
    data['accel_direction'] = np.sign(data['acceleration'])
    data['persistence_count'] = 0
    for i in range(1, len(data)):
        if data['accel_direction'].iloc[i] == data['accel_direction'].iloc[i-1]:
            data['persistence_count'].iloc[i] = min(data['persistence_count'].iloc[i-1] + 1, 5)
        else:
            data['persistence_count'].iloc[i] = 1
    
    # Volatility-Adjusted Acceleration
    data['daily_range'] = data['high'] - data['low']
    data['range_vol_5d'] = data['daily_range'].rolling(window=5).mean()
    data['vol_scaled_accel'] = (np.cbrt(data['acceleration'] * data['persistence_count']) / 
                               data['range_vol_5d'].replace(0, np.nan))
    
    # Volume Acceleration System
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_accel_3d'] = (data['volume_momentum_5d'] / 
                              data['volume_momentum_5d'].shift(3).replace(0, np.nan)) - 1
    
    # Divergence Classification and Strength
    data['divergence_type'] = 0
    data.loc[(data['acceleration'] > 0) & (data['volume_accel_3d'] < 0), 'divergence_type'] = 1  # Positive
    data.loc[(data['acceleration'] < 0) & (data['volume_accel_3d'] > 0), 'divergence_type'] = -1  # Negative
    
    data['divergence_magnitude'] = abs(data['acceleration'] - data['volume_accel_3d'])
    data['divergence_duration'] = 0
    for i in range(1, len(data)):
        if data['divergence_type'].iloc[i] == data['divergence_type'].iloc[i-1]:
            data['divergence_duration'].iloc[i] = data['divergence_duration'].iloc[i-1] + 1
        else:
            data['divergence_duration'].iloc[i] = 1
    
    data['divergence_signal'] = data['divergence_magnitude'] * data['divergence_duration']
    
    # Intraday Gap Confirmation System
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    data['intraday_realization'] = data['close'] / data['open'] - 1
    
    data['gap_persistence'] = 0
    for i in range(1, len(data)):
        if np.sign(data['overnight_gap'].iloc[i]) == np.sign(data['overnight_gap'].iloc[i-1]):
            data['gap_persistence'].iloc[i] = data['gap_persistence'].iloc[i-1] + 1
        else:
            data['gap_persistence'].iloc[i] = 1
    
    data['gap_persistence_score'] = data['gap_persistence'] * abs(data['overnight_gap'])
    data['gap_alignment_multiplier'] = np.sign(data['overnight_gap']) * np.sign(data['acceleration'])
    
    # Intraday Efficiency Confirmation
    data['intraday_range_efficiency'] = (data['high'] - data['low']) / abs(data['open'] - data['close']).replace(0, np.nan)
    data['volume_trend_alignment'] = np.sign(data['volume'] - data['volume'].shift(1)) * np.sign(data['close'] - data['close'].shift(1))
    data['intraday_confirmation_score'] = data['intraday_range_efficiency'] * data['volume_trend_alignment']
    
    # Volume-Confirmed Signal Integration
    data['range_liquidity'] = ((data['high'] - data['low']) / data['close']) * data['volume']
    data['volume_ma_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_ma_10d'] = data['volume'].rolling(window=10).mean()
    data['volume_trend'] = (data['volume_ma_5d'] / data['volume_ma_10d']) - 1
    data['volume_confirmation_filter'] = np.cbrt(data['range_liquidity'] * data['volume_trend'])
    
    # Range Expansion Detection
    data['avg_range_5d'] = data['daily_range'].rolling(window=5).mean()
    data['range_expansion_factor'] = data['daily_range'] / data['avg_range_5d'].replace(0, np.nan)
    
    # Composite Factor Synthesis
    data['accel_divergence_base'] = data['vol_scaled_accel'] * data['divergence_signal']
    data['gap_confirmation_multiplier'] = data['gap_alignment_multiplier'] * data['gap_persistence_score']
    data['intraday_boost'] = data['intraday_confirmation_score'] * data['range_expansion_factor']
    
    # Persistence Requirements
    data['acceleration_consistent'] = data['persistence_count'] >= 2
    data['divergence_sustained'] = data['divergence_duration'] >= 2
    
    # Final Factor Generation
    data['volume_weighted_core'] = data['accel_divergence_base'] * data['volume_confirmation_filter']
    data['confirmation_enhancement'] = data['volume_weighted_core'] * data['gap_confirmation_multiplier']
    
    # Apply persistence filters
    mask = data['acceleration_consistent'] & data['divergence_sustained']
    data['final_factor'] = data['confirmation_enhancement'] * data['intraday_boost']
    data.loc[~mask, 'final_factor'] = 0
    
    return data['final_factor']
