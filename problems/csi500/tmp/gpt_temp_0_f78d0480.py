import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate intermediate momentum
    # 5-day return
    ret_5d = df['close'] / df['close'].shift(5) - 1
    
    # 20-day return
    ret_20d = df['close'] / df['close'].shift(20) - 1
    
    # Combine returns and take cube root
    momentum_combined = (ret_5d * ret_20d).replace([np.inf, -np.inf], np.nan)
    intermediate_momentum = np.cbrt(momentum_combined)
    
    # Calculate volatility regime
    # Daily range
    daily_range = df['high'] - df['low']
    
    # 10-day average range
    avg_range_10d = daily_range.rolling(window=10, min_periods=5).mean()
    
    # Normalize range by close price and multiply by 100
    normalized_vol = (avg_range_10d / df['close']) * 100
    
    # Calculate 60-day volatility percentile
    vol_percentile = normalized_vol.rolling(window=60, min_periods=30).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    
    # Center percentile around zero
    regime_adjustment = vol_percentile - 0.5
    
    # Apply non-linear scaling with hyperbolic tangent
    factor = intermediate_momentum * np.tanh(regime_adjustment * 2)
    
    return factor
