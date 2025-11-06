import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Decay Ratio
    short_term_momentum = data['close'] / data['close'].shift(3) - 1
    medium_term_momentum = data['close'] / data['close'].shift(10) - 1
    momentum_decay_ratio = short_term_momentum / (medium_term_momentum + 1e-6) - 1
    
    # Acceleration Asymmetry
    accel_2d = data['close'] / data['close'].shift(2) - data['close'].shift(2) / data['close'].shift(4)
    positive_accel = np.maximum(0, accel_2d)
    negative_accel = np.minimum(0, accel_2d)
    asymmetry_ratio = positive_accel / (negative_accel - 1e-6)
    
    # Volume Pressure Differential
    up_days = data['close'] > data['close'].shift(1)
    down_days = data['close'] < data['close'].shift(1)
    
    up_volume_5d = data['volume'].rolling(window=5).apply(
        lambda x: np.sum(x[up_days.iloc[-len(x):].values]) if len(x) == 5 else np.nan, raw=False
    )
    up_volume_20d = data['volume'].rolling(window=20).apply(
        lambda x: np.sum(x[up_days.iloc[-len(x):].values]) if len(x) == 20 else np.nan, raw=False
    )
    down_volume_5d = data['volume'].rolling(window=5).apply(
        lambda x: np.sum(x[down_days.iloc[-len(x):].values]) if len(x) == 5 else np.nan, raw=False
    )
    down_volume_20d = data['volume'].rolling(window=20).apply(
        lambda x: np.sum(x[down_days.iloc[-len(x):].values]) if len(x) == 20 else np.nan, raw=False
    )
    
    up_pressure = up_volume_5d / (up_volume_20d + 1e-6)
    down_pressure = down_volume_5d / (down_volume_20d + 1e-6)
    volume_pressure_diff = up_pressure - down_pressure
    
    # True Range Efficiency
    true_range_efficiency = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-6)
    
    # Composite Alpha
    core_signal = momentum_decay_ratio * true_range_efficiency * asymmetry_ratio
    final_alpha = core_signal * volume_pressure_diff * np.sign(momentum_decay_ratio)
    
    return final_alpha
