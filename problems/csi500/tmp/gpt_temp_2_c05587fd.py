import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Acceleration
    short_price_change = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    medium_price_change = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    price_acceleration = short_price_change / medium_price_change
    
    # Calculate Volume Acceleration
    short_volume_change = (df['volume'] - df['volume'].shift(3)) / df['volume'].shift(3)
    medium_volume_change = (df['volume'] - df['volume'].shift(8)) / df['volume'].shift(8)
    volume_acceleration = short_volume_change / medium_volume_change
    
    # Compute Asymmetric Volatility Adjustment
    daily_returns = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    upside_volatility = daily_returns.rolling(window=7, min_periods=1).apply(
        lambda x: x[x > 0].std() if len(x[x > 0]) > 1 else 0, raw=False
    )
    
    downside_volatility = daily_returns.rolling(window=7, min_periods=1).apply(
        lambda x: x[x < 0].std() if len(x[x < 0]) > 1 else 0, raw=False
    )
    
    volatility_asymmetry = upside_volatility - downside_volatility
    
    # Generate Alpha Factor
    raw_factor = price_acceleration - volume_acceleration
    volatility_multiplier = 1 + volatility_asymmetry * np.sign(raw_factor)
    final_factor = raw_factor * volatility_multiplier * (1 + np.abs(raw_factor))
    
    return final_factor
