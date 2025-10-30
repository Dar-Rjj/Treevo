import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Component
    # Short-Term Volatility
    df['high_5d'] = df['high'].rolling(window=5).max()
    df['low_5d'] = df['low'].rolling(window=5).min()
    df['volatility_5d_range'] = (df['high_5d'] - df['low_5d']) / df['close'].shift(4)
    
    df['volatility_10d_var'] = df['close'].rolling(window=10).var()
    
    # Volatility Regime Detection
    df['high_20d'] = df['high'].rolling(window=20).max()
    df['low_20d'] = df['low'].rolling(window=20).min()
    df['volatility_20d_range'] = (df['high_20d'] - df['low_20d']) / df['close'].shift(19)
    
    df['volatility_regime'] = df['volatility_5d_range'] / df['volatility_20d_range']
    df['volatility_acceleration'] = df['volatility_5d_range'] - df['volatility_5d_range'].shift(1)
    df['volatility_weight'] = df['volatility_regime'] * df['volatility_acceleration']
    
    # Price-Volume Acceleration
    # Price Acceleration
    df['return_3d_recent'] = df['close'] / df['close'].shift(2) - 1
    df['return_3d_prior'] = df['close'].shift(2) / df['close'].shift(5) - 1
    df['price_acceleration_5d'] = df['return_3d_recent'] - df['return_3d_prior']
    
    df['return_5d_recent'] = df['close'] / df['close'].shift(4) - 1
    df['return_5d_prior'] = df['close'].shift(4) / df['close'].shift(9) - 1
    df['price_curvature_10d'] = df['return_5d_recent'] - df['return_5d_prior']
    
    # Price Change Persistence
    df['daily_return'] = df['close'].pct_change()
    df['return_sign'] = np.sign(df['daily_return'])
    
    def count_consecutive_signs(series):
        signs = series.dropna()
        if len(signs) == 0:
            return 0
        current_sign = signs.iloc[-1]
        count = 1
        for i in range(len(signs)-2, -1, -1):
            if signs.iloc[i] == current_sign:
                count += 1
            else:
                break
        return count
    
    df['consecutive_days'] = df['return_sign'].rolling(window=10).apply(
        count_consecutive_signs, raw=False
    )
    
    df['acceleration_decay'] = df['price_acceleration_5d'].abs() / (
        df['price_acceleration_5d'].abs().rolling(window=10).mean() + 1e-8
    )
    
    # Volume Acceleration
    df['volume_change_5d'] = df['volume'] / (df['volume'].shift(5) + 1e-8)
    
    df['volume_avg_recent'] = df['volume'].rolling(window=3).mean()
    df['volume_avg_prior'] = df['volume'].shift(3).rolling(window=3).mean()
    df['volume_acceleration'] = df['volume_avg_recent'] / (df['volume_avg_prior'] + 1e-8) - 1
    
    # Price-Volume Acceleration Correlation
    df['price_volume_direction'] = np.sign(df['price_acceleration_5d']) * np.sign(df['volume_acceleration'])
    df['acceleration_magnitude_alignment'] = (
        df['price_acceleration_5d'].abs() * df['volume_acceleration'].abs()
    )
    
    # Dynamic Signal Integration
    # Volatility-Weighted Acceleration
    df['vol_weighted_price_accel'] = df['price_acceleration_5d'] / (df['volatility_5d_range'] + 1e-8)
    df['scaled_volume_accel'] = df['volume_acceleration'] * df['volatility_regime']
    
    # Convergence-Divergence Detection
    df['acceleration_convergence'] = (
        (df['price_volume_direction'] > 0) & 
        (df['volatility_weight'] > 0)
    ).astype(float)
    
    df['reversal_anticipation'] = (
        df['acceleration_convergence'] * 
        df['acceleration_magnitude_alignment'] * 
        (1 - df['volatility_regime'].clip(0, 1)) * 
        df['consecutive_days'] / 10
    )
    
    # Final factor calculation
    factor = (
        df['vol_weighted_price_accel'] * 0.4 +
        df['scaled_volume_accel'] * 0.3 +
        df['reversal_anticipation'] * 0.3 +
        df['price_curvature_10d'] * 0.2
    )
    
    return factor
