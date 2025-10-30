import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volatility-Adjusted Momentum Divergence factor that combines price momentum,
    volatility normalization, volume confirmation, and price range validation.
    """
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    # Calculate Price Momentum
    momentum_mean = close.shift(1).rolling(window=20, min_periods=10).mean()
    momentum_deviation = (close - momentum_mean) / momentum_mean
    
    # Calculate Volatility Adjustment
    volatility = close.shift(1).rolling(window=20, min_periods=10).std()
    volatility_adjusted_momentum = momentum_deviation / (volatility + 1e-8)
    
    # Volume Confirmation
    volume_slope = volume.rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
    )
    volume_confirmed_momentum = volatility_adjusted_momentum * volume_slope
    
    # Price Range Validation
    daily_range = (high - low) / close
    range_momentum_consistency = np.sign(volume_confirmed_momentum) * np.sign(momentum_deviation)
    range_adjustment = np.where(range_momentum_consistency > 0, 1 + daily_range, 1 - daily_range)
    
    # Final factor combining all components
    factor = volume_confirmed_momentum * range_adjustment
    
    return factor
