import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum acceleration with volatility-normalized smoothing and volume-confirmed efficiency
    # Uses nonlinear transforms for regime-aware signal enhancement across price-volume trends
    
    # Momentum acceleration: difference between short-term and medium-term momentum
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_8d = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    momentum_acceleration = momentum_3d - momentum_8d
    
    # Volatility-normalized smoothing using rolling median absolute deviation
    returns = df['close'].pct_change()
    volatility_mad = returns.rolling(window=5).apply(lambda x: (x - x.median()).abs().median())
    
    # Nonlinear volatility adjustment using sigmoid transform for regime awareness
    volatility_normalized = 2 / (1 + np.exp(-volatility_mad * 10)) - 1
    
    # Volume-confirmed efficiency with trend alignment
    volume_trend = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    price_trend = df['close'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Volume-price trend alignment using correlation-like measure
    trend_alignment = np.sign(volume_trend * price_trend) * (abs(volume_trend * price_trend) ** 0.5)
    
    # Price efficiency with high-low range normalization
    daily_range = df['high'] - df['low']
    efficiency_ratio = abs(df['close'] - df['close'].shift(1)) / (daily_range + 1e-7)
    
    # Combine components with nonlinear interactions for robustness
    alpha_factor = (
        momentum_acceleration * 
        (1 + volatility_normalized) * 
        np.tanh(trend_alignment * 2) * 
        np.sqrt(efficiency_ratio)
    )
    
    return alpha_factor
