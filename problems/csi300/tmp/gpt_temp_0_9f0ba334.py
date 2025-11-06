import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate price returns
    data['price_return'] = data['close'].pct_change()
    
    # Calculate first-order price acceleration
    data['price_accel_1'] = data['price_return'].diff()
    
    # Calculate second-order price acceleration (3-day window)
    data['price_accel_2'] = data['price_accel_1'].rolling(window=3, min_periods=2).mean()
    
    # Calculate third-order price acceleration (5-day window)
    data['price_accel_3'] = data['price_accel_2'].rolling(window=5, min_periods=3).mean()
    
    # Calculate volume returns
    data['volume_return'] = data['volume'].pct_change()
    
    # Calculate first-order volume acceleration
    data['volume_accel_1'] = data['volume_return'].diff()
    
    # Calculate second-order volume acceleration (3-day window)
    data['volume_accel_2'] = data['volume_accel_1'].rolling(window=3, min_periods=2).mean()
    
    # Calculate third-order volume acceleration (5-day window)
    data['volume_accel_3'] = data['volume_accel_2'].rolling(window=5, min_periods=3).mean()
    
    # Directional asymmetry analysis
    data['dir_asymmetry'] = np.where(
        np.sign(data['price_accel_1']) != np.sign(data['volume_accel_1']),
        np.abs(data['price_accel_1'] - data['volume_accel_1']),
        0
    )
    
    # Magnitude asymmetry analysis
    data['magnitude_asymmetry'] = (
        np.abs(data['price_accel_1']) - np.abs(data['volume_accel_1'])
    ) / (np.abs(data['price_accel_1']) + np.abs(data['volume_accel_1']) + 1e-8)
    
    # Regime identification
    data['price_regime'] = np.sign(data['price_accel_1'].rolling(window=5, min_periods=3).mean())
    data['volume_regime'] = np.sign(data['volume_accel_1'].rolling(window=5, min_periods=3).mean())
    data['regime_sync'] = data['price_regime'] * data['volume_regime']
    
    # Acceleration persistence
    data['price_persistence'] = (
        data['price_accel_1'].rolling(window=5, min_periods=3).apply(
            lambda x: len([i for i in range(1, len(x)) if np.sign(x.iloc[i]) == np.sign(x.iloc[i-1])]) / max(len(x)-1, 1)
        )
    )
    
    data['volume_persistence'] = (
        data['volume_accel_1'].rolling(window=5, min_periods=3).apply(
            lambda x: len([i for i in range(1, len(x)) if np.sign(x.iloc[i]) == np.sign(x.iloc[i-1])]) / max(len(x)-1, 1)
        )
    )
    
    # Asymmetry persistence
    data['asymmetry_persistence'] = (
        data['dir_asymmetry'].rolling(window=5, min_periods=3).apply(
            lambda x: np.mean(x > 0) if len(x) > 0 else 0
        )
    )
    
    # Generate final signal with weighted components
    # Base signal from first-order accelerations
    base_signal = (
        0.5 * data['price_accel_1'] + 
        0.3 * data['price_accel_2'] + 
        0.2 * data['price_accel_3']
    )
    
    # Apply directional asymmetry multiplier
    dir_multiplier = 1 + 2 * data['dir_asymmetry'].fillna(0)
    base_signal = base_signal * dir_multiplier
    
    # Apply magnitude asymmetry adjustment
    mag_adjustment = 1 + data['magnitude_asymmetry'].fillna(0)
    base_signal = base_signal * mag_adjustment
    
    # Regime-based filtering
    regime_weight = np.where(
        data['regime_sync'] > 0, 
        1.2,  # Enhanced signal during synchronized regimes
        np.where(data['regime_sync'] < 0, 0.8, 1.0)  # Reduced signal during conflicting regimes
    )
    base_signal = base_signal * regime_weight
    
    # Persistence adjustment
    persistence_score = (
        0.6 * data['price_persistence'].fillna(0.5) + 
        0.4 * data['volume_persistence'].fillna(0.5)
    )
    base_signal = base_signal * persistence_score
    
    # Final asymmetry pattern stability weighting
    stability_weight = 0.5 + 0.5 * data['asymmetry_persistence'].fillna(0.5)
    final_signal = base_signal * stability_weight
    
    # Return the final factor values
    return final_signal
