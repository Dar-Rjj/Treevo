import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Behavioral Momentum Acceleration Components
    # Round Number Acceleration
    data['price_change'] = data['close'] / data['close'].shift(1) - 1
    data['price_acceleration'] = data['price_change'] - data['price_change'].shift(1)
    data['round_proximity'] = 1 - abs(data['close'] - np.round(data['close'])) / data['close']
    data['round_acceleration'] = data['price_acceleration'] * data['round_proximity']
    
    # Extreme Proximity Acceleration
    data['high_20d'] = data['high'].rolling(window=20, min_periods=1).max()
    data['low_20d'] = data['low'].rolling(window=20, min_periods=1).min()
    data['high_proximity'] = (data['high'] - data['high_20d']) / data['high']
    data['low_proximity'] = (data['low_20d'] - data['low']) / data['low']
    data['high_extreme_acceleration'] = data['price_acceleration'] * data['high_proximity']
    data['low_extreme_acceleration'] = data['price_acceleration'] * data['low_proximity']
    
    # Volatility-Regime Adaptive Framework
    data['current_volatility'] = data['high'] - data['low']
    data['historical_volatility'] = (data['high'] - data['low']).rolling(window=5, min_periods=1).median()
    data['regime_ratio'] = data['current_volatility'] / data['historical_volatility'].replace(0, np.nan)
    
    # Volume-Price Co-movement Dynamics
    data['price_intensity'] = abs(data['price_change'])
    data['volume_change'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_intensity'] = abs(data['volume_change'])
    data['price_direction'] = np.sign(data['close'] - data['close'].shift(1))
    data['co_movement'] = data['price_intensity'] * data['volume_intensity'] * data['price_direction']
    
    data['amount_ma'] = data['amount'].rolling(window=5, min_periods=1).mean()
    data['amount_momentum_density'] = data['price_change'] * (data['amount'] / data['amount_ma'])
    
    # Microstructure Acceleration Patterns
    data['spread_acceleration_vol'] = ((data['high'] - data['low']) / data['close']) * data['price_acceleration']
    
    # Volume-weighted acceleration flow (3-day window)
    vol_weighted_accel = []
    for i in range(len(data)):
        if i >= 3:
            window_data = data.iloc[i-3:i+1]
            accel_contrib = []
            for j in range(1, 4):
                if i-j >= 0 and i-j-1 >= 0 and i-j-2 >= 0:
                    price_change_curr = window_data['close'].iloc[j] / window_data['close'].iloc[j-1] - 1
                    price_change_prev = window_data['close'].iloc[j-1] / window_data['close'].iloc[j-2] - 1
                    acceleration = price_change_curr - price_change_prev
                    accel_contrib.append(acceleration * window_data['volume'].iloc[j])
            vol_weighted_accel.append(sum(accel_contrib) if accel_contrib else 0)
        else:
            vol_weighted_accel.append(0)
    data['vol_weighted_accel_flow'] = vol_weighted_accel
    
    # Behavioral-Volume Acceleration Resonance
    data['round_accel_volume'] = data['round_acceleration'] * data['volume']
    data['high_extreme_accel_volume'] = data['high_extreme_acceleration'] * data['volume']
    data['low_extreme_accel_volume'] = data['low_extreme_acceleration'] * data['volume']
    
    # Adaptive Acceleration Convergence Synthesis
    # Normalize acceleration components
    acceleration_components = ['round_acceleration', 'high_extreme_acceleration', 'low_extreme_acceleration']
    for col in acceleration_components:
        data[f'{col}_norm'] = (data[col] - data[col].rolling(window=20, min_periods=1).mean()) / data[col].rolling(window=20, min_periods=1).std().replace(0, 1)
    
    # Regime-weighted acceleration combination
    data['behavioral_acceleration'] = (
        data['round_acceleration_norm'] * 0.4 + 
        data['high_extreme_acceleration_norm'] * 0.3 + 
        data['low_extreme_acceleration_norm'] * 0.3
    )
    
    # Final Alpha Construction
    # Apply regime-dependent weighting
    high_vol_threshold = data['regime_ratio'].quantile(0.7)
    low_vol_threshold = data['regime_ratio'].quantile(0.3)
    
    def regime_signal(row):
        if row['regime_ratio'] > high_vol_threshold:
            return row['behavioral_acceleration'] * row['co_movement'] * row['regime_ratio']
        elif row['regime_ratio'] < low_vol_threshold:
            return row['behavioral_acceleration'] * row['co_movement'] / max(row['regime_ratio'], 0.1)
        else:
            return row['behavioral_acceleration'] * row['co_movement']
    
    data['alpha_signal'] = data.apply(regime_signal, axis=1)
    
    # Apply bounded transformation for factor stability
    alpha_series = data['alpha_signal'].copy()
    
    # Remove extreme outliers and normalize
    alpha_series = alpha_series.clip(lower=alpha_series.quantile(0.01), upper=alpha_series.quantile(0.99))
    alpha_series = (alpha_series - alpha_series.rolling(window=20, min_periods=1).mean()) / alpha_series.rolling(window=20, min_periods=1).std().replace(0, 1)
    
    return alpha_series
