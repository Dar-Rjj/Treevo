import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Price acceleration with volume efficiency and volatility normalization
    # Focuses on acceleration patterns, trading efficiency, and cleaner volatility handling
    
    # Price acceleration (rate of change of momentum)
    momentum_3d = df['close'] / df['close'].shift(3) - 1
    momentum_6d = df['close'] / df['close'].shift(6) - 1
    price_acceleration = momentum_3d - momentum_6d
    
    # Volume efficiency (volume per unit of price movement)
    price_range = df['high'] - df['low']
    volume_efficiency = df['volume'] / (price_range + 1e-7)
    
    # Trading range efficiency (how efficiently price moves from open to close)
    daily_movement = abs(df['close'] - df['open'])
    range_utilization = daily_movement / (price_range + 1e-7)
    
    # Volatility normalization using median absolute deviation (cleaner than std)
    returns = df['close'].pct_change()
    volatility_mad = returns.rolling(window=10, min_periods=10).apply(lambda x: abs(x - x.median()).median())
    
    # Combine acceleration with efficiency metrics
    acceleration_efficiency = price_acceleration * volume_efficiency * range_utilization
    
    # Apply robust volatility normalization
    volatility_normalized_factor = acceleration_efficiency / (volatility_mad + 1e-7)
    
    return volatility_normalized_factor
