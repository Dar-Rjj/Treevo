import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum Decay Factor with volume acceleration and volatility scaling
    """
    # Calculate Intraday Momentum
    intraday_momentum = ((df['high'] + df['low']) / 2 - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Price Range Volatility with exponential smoothing
    price_range_volatility = (df['high'] - df['low']) / df['close'].shift(1)
    vol_ewm = price_range_volatility.ewm(alpha=0.1, adjust=False).mean()
    
    # Calculate Momentum Persistence (sign consistency over 3 days)
    momentum_sign = np.sign(intraday_momentum)
    momentum_persistence = momentum_sign.rolling(window=3).apply(
        lambda x: len(set(x.dropna())) == 1 if len(x.dropna()) == 3 else 0, raw=False
    )
    
    # Calculate Volume Trend with exponential smoothing
    volume_trend = (df['volume'] / df['volume'].shift(1) - 1).ewm(alpha=0.15, adjust=False).mean()
    
    # Combine factors
    raw_factor = intraday_momentum * volume_trend * momentum_persistence
    final_factor = raw_factor / vol_ewm
    
    # Handle NaN values
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return final_factor
