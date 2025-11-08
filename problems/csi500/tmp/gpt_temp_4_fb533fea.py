import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize epsilon for numerical stability
    epsilon = 1e-8
    
    # Extract price and volume data
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Volatility Regime Classification
    daily_range = high - low
    rolling_volatility = daily_range.rolling(window=20).std()
    
    # Percentile-based regime classification
    volatility_percentile = rolling_volatility.rolling(window=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Create regime weights
    regime_weight = np.where(volatility_percentile > 0.8, 1.2,
                            np.where(volatility_percentile < 0.2, 0.8, 1.0))
    
    # Multi-Timeframe Volatility-Adjusted Momentum
    # Short-term momentum (3-day)
    short_return = close / close.shift(3) - 1
    short_momentum = short_return / (rolling_volatility + epsilon)
    
    # Medium-term momentum (8-day)
    medium_return = close / close.shift(8) - 1
    medium_momentum = medium_return / (rolling_volatility + epsilon)
    
    # Long-term momentum (20-day)
    long_return = close / close.shift(20) - 1
    long_momentum = long_return / (rolling_volatility + epsilon)
    
    # Multiplicative Momentum Integration
    momentum_product = short_momentum * medium_momentum * long_momentum
    momentum_cube_root = np.sign(momentum_product) * np.abs(momentum_product) ** (1/3)
    
    # Volume-Price Divergence Signal
    # Price trend strength
    price_momentum = close / close.shift(5) - 1
    price_trend_strength = np.abs(price_momentum)
    
    # Volume trend strength
    volume_momentum = volume / volume.shift(5) - 1
    volume_trend_strength = np.abs(volume_momentum)
    
    # Divergence ratio calculation
    divergence_ratio = price_trend_strength / (volume_trend_strength + epsilon)
    divergence_cube_root = np.sign(divergence_ratio) * np.abs(divergence_ratio) ** (1/3)
    
    # Final Alpha Construction
    base_factor = regime_weight * momentum_cube_root * divergence_cube_root
    
    # Signal bounding
    bounded_factor = np.clip(base_factor, -2.0, 2.0)
    
    return pd.Series(bounded_factor, index=df.index)
