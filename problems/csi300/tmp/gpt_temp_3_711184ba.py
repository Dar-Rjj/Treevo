import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Acceleration Calculation
    # Short-Term Momentum
    momentum_5d = data['close'] / data['close'].shift(5) - 1
    momentum_3d = data['close'] / data['close'].shift(3) - 1
    
    # Momentum Acceleration Signal
    acceleration = momentum_3d - momentum_5d
    acceleration_ratio = acceleration / (np.abs(momentum_5d) + 1e-8)
    
    # Volatility Adjustment
    # Daily True Range Calculation
    prev_close = data['close'].shift(1)
    tr1 = data['high'] - data['low']
    tr2 = np.abs(data['high'] - prev_close)
    tr3 = np.abs(data['low'] - prev_close)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # 10-day Average True Range
    atr_10d = true_range.rolling(window=10, min_periods=1).mean()
    
    # Volatility Scaling
    volatility_weighted_acceleration = acceleration_ratio / (atr_10d + 1e-8)
    
    # Volume-Price Divergence
    # Price Momentum Component
    price_change_5d = data['close'] / data['close'].shift(5) - 1
    price_direction = np.sign(price_change_5d)
    
    # Volume Momentum Component
    volume_change_5d = data['volume'] / data['volume'].shift(5) - 1
    volume_ma_5d = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_ma_10d = data['volume'].rolling(window=10, min_periods=1).mean()
    volume_ratio_5d_10d = volume_ma_5d / volume_ma_10d - 1
    
    # Volume-Price Divergence Calculation
    volume_momentum = volume_change_5d * volume_ratio_5d_10d
    volume_direction = np.sign(volume_momentum)
    
    # Detect divergence and calculate magnitude
    divergence_sign = np.where(price_direction != volume_direction, -1, 1)
    divergence_magnitude = np.abs(price_change_5d * volume_momentum)
    volume_price_divergence = divergence_sign * divergence_magnitude
    
    # Amount-based Confirmation
    amount_trend = data['amount'].rolling(window=5, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
    )
    amount_direction = np.sign(amount_trend)
    
    # Combined Alpha Factor
    alpha_factor = (volatility_weighted_acceleration * 
                   volume_price_divergence * 
                   amount_direction)
    
    return alpha_factor
