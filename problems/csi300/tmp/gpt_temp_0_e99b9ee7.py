import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Weighted Price Acceleration Divergence
    # Calculate Price Acceleration
    price_3d_return = df['close'].pct_change(3)
    price_5d_return = df['close'].pct_change(5)
    price_acceleration = price_3d_return - price_5d_return
    
    # Compute Volume-Weighted Acceleration
    volume_ratio = df['volume'] / df['volume'].rolling(window=10, min_periods=1).mean()
    weighted_acceleration = price_acceleration * volume_ratio
    
    # Detect Divergence Patterns
    price_trend = df['close'].rolling(window=10, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0, raw=False
    )
    acceleration_direction = np.sign(price_acceleration)
    trend_direction = np.sign(price_trend)
    divergence_strength = np.where(
        acceleration_direction != trend_direction,
        np.abs(price_acceleration) * np.abs(price_trend),
        0
    )
    
    # Generate Alpha Signal
    recent_volatility = df['close'].pct_change().rolling(window=10, min_periods=1).std()
    alpha_signal = weighted_acceleration * divergence_strength / (recent_volatility + 1e-8)
    
    return alpha_signal
