import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Efficiency Component
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    df['true_range_efficiency'] = abs(df['close'] - df['prev_close']) / df['true_range']
    df['true_range_efficiency'] = df['true_range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    df['efficiency_3d_avg'] = df['true_range_efficiency'].rolling(window=3, min_periods=1).mean()
    df['efficiency_8d_avg'] = df['true_range_efficiency'].rolling(window=8, min_periods=1).mean()
    df['efficiency_momentum'] = df['efficiency_3d_avg'] - df['efficiency_8d_avg']
    df['efficiency_volatility'] = df['true_range_efficiency'].rolling(window=5, min_periods=1).std()
    
    # Volume-Price Acceleration Divergence
    df['price_accel'] = (df['close'] / df['close'].shift(3) - 1) - (df['close'] / df['close'].shift(8) - 1)
    df['volume_accel'] = (df['volume'] / df['volume'].shift(3) - 1) - (df['volume'] / df['volume'].shift(8) - 1)
    df['accel_divergence'] = df['price_accel'] - df['volume_accel']
    
    df['accel_div_sign'] = np.sign(df['accel_divergence'])
    df['same_sign_count'] = 0
    for i in range(1, len(df)):
        if df['accel_div_sign'].iloc[i] == df['accel_div_sign'].iloc[i-1]:
            df['same_sign_count'].iloc[i] = df['same_sign_count'].iloc[i-1] + 1
        else:
            df['same_sign_count'].iloc[i] = 1
    df['divergence_persistence'] = df['accel_div_sign'] * df['same_sign_count'].rolling(window=5, min_periods=1).max()
    
    # Trade Size Dynamics
    df['trade_size'] = df['amount'] / df['volume']
    df['trade_size'] = df['trade_size'].replace([np.inf, -np.inf], np.nan)
    df['avg_trade_size_3d'] = df['trade_size'].rolling(window=3, min_periods=1).mean()
    df['trade_size_ratio'] = df['trade_size'] / df['avg_trade_size_3d']
    df['trade_size_volatility'] = df['trade_size_ratio'].rolling(window=5, min_periods=1).std()
    df['size_efficiency_corr'] = df['trade_size_ratio'].rolling(window=5, min_periods=1).corr(df['true_range_efficiency'])
    
    # Range-Based Momentum Filter
    df['daily_range'] = df['high'] - df['low']
    df['avg_range_3d'] = df['daily_range'].rolling(window=3, min_periods=1).mean()
    df['range_expansion'] = df['daily_range'] / df['avg_range_3d']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['close_position'] = df['close_position'].replace([np.inf, -np.inf], np.nan)
    df['range_momentum'] = df['range_expansion'] * df['close_position']
    
    # Directional Filtered Alpha
    conditions = [
        (df['price_accel'] > 0) & (df['volume_accel'] < 0) & (df['range_momentum'] > 0.5),
        (df['price_accel'] < 0) & (df['volume_accel'] > 0) & (df['range_momentum'] < 0.5),
        (df['divergence_persistence'] > 2) & (df['size_efficiency_corr'] > 0),
        (df['divergence_persistence'] < -2) & (df['size_efficiency_corr'] < 0)
    ]
    choices = [1, -1, 0.5, -0.5]
    df['multiplier'] = np.select(conditions, choices, default=1.0)
    
    # Final Alpha
    alpha = (df['efficiency_momentum'] * df['trade_size_ratio'] + 
             df['accel_divergence'] * df['efficiency_volatility']) * df['multiplier']
    
    return alpha
