import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Acceleration Components
    # First-Order Price Momentum
    data['momentum_1'] = data['close'] / data['close'].shift(1) - 1
    
    # Second-Order Price Acceleration
    data['acceleration_2'] = data['momentum_1'] - data['momentum_1'].shift(1)
    
    # Third-Order Price Jerk
    data['jerk_3'] = data['acceleration_2'] - data['acceleration_2'].shift(1)
    
    # Volume Confirmation Signals
    # Volume Trend (4-day lookback)
    data['volume_trend'] = data['volume'] / data['volume'].shift(4) - 1
    
    # Volume Acceleration
    data['volume_acceleration'] = data['volume_trend'] - data['volume_trend'].shift(1)
    
    # Generate Volume Confirmation Flags
    data['positive_accel_volume_up'] = ((data['acceleration_2'] > 0) & (data['volume_acceleration'] > 0)).astype(int)
    data['negative_accel_volume_down'] = ((data['acceleration_2'] < 0) & (data['volume_acceleration'] < 0)).astype(int)
    data['price_volume_divergence'] = ((data['acceleration_2'] > 0) & (data['volume_acceleration'] < 0)) | \
                                     ((data['acceleration_2'] < 0) & (data['volume_acceleration'] > 0))
    data['divergence_flag'] = data['price_volume_divergence'].astype(int)
    
    # Market Regime Changes
    # Daily High-Low Range
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    
    # Regime Volatility Threshold (20-day median)
    data['regime_vol_threshold'] = data['daily_range'].rolling(window=20, min_periods=10).median()
    
    # Identify Regime Shifts
    data['high_vol_regime'] = (data['daily_range'] > data['regime_vol_threshold']).astype(int)
    data['low_vol_regime'] = (data['daily_range'] <= data['regime_vol_threshold']).astype(int)
    
    # Calculate Regime Persistence
    regime_changes = data['high_vol_regime'].diff().fillna(0) != 0
    data['regime_persistence'] = 0
    current_streak = 0
    
    for i in range(len(data)):
        if i == 0:
            data.iloc[i, data.columns.get_loc('regime_persistence')] = 1
            current_streak = 1
        else:
            if not regime_changes.iloc[i]:
                current_streak += 1
            else:
                current_streak = 1
            data.iloc[i, data.columns.get_loc('regime_persistence')] = current_streak
    
    # Generate Composite Alpha Factor
    # Combine Acceleration Components with weights
    acceleration_composite = (
        0.4 * data['momentum_1'].fillna(0) +
        0.35 * data['acceleration_2'].fillna(0) +
        0.25 * data['jerk_3'].fillna(0)
    )
    
    # Apply Volume Confirmation Weights
    volume_multiplier = np.where(
        data['positive_accel_volume_up'] == 1, 1.2,
        np.where(
            data['negative_accel_volume_down'] == 1, 1.1,
            np.where(
                data['divergence_flag'] == 1, 0.7,
                1.0
            )
        )
    )
    
    # Adjust for Regime Context
    regime_adjustment = np.where(
        data['high_vol_regime'] == 1,
        0.8 + (0.1 * np.minimum(data['regime_persistence'] / 10, 1)),
        np.where(
            data['low_vol_regime'] == 1,
            1.1 + (0.05 * np.minimum(data['regime_persistence'] / 20, 1)),
            1.0
        )
    )
    
    # Final composite factor
    alpha_factor = acceleration_composite * volume_multiplier * regime_adjustment
    
    # Return as pandas Series with same index as input
    return pd.Series(alpha_factor, index=data.index, name='price_acceleration_volume_regime_factor')
