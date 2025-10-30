import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Asymmetric Volatility-Adjusted Momentum factor
    Combines short and medium-term momentum with volatility asymmetry adjustment
    """
    close = data['close']
    
    # Compute Momentum components
    momentum_10d = close.pct_change(10)
    momentum_20d = close.pct_change(20)
    
    # Calculate daily returns for volatility measurement
    returns = close.pct_change()
    
    # Measure Volatility Asymmetry
    upside_vol = returns.rolling(window=20).apply(
        lambda x: x[x > 0].std() if len(x[x > 0]) > 1 else np.nan
    )
    
    downside_vol = returns.rolling(window=20).apply(
        lambda x: x[x < 0].std() if len(x[x < 0]) > 1 else np.nan
    )
    
    # Calculate Volatility Adjustment Ratio
    vol_ratio = upside_vol / downside_vol
    vol_adjustment = np.log(vol_ratio.replace([np.inf, -np.inf], np.nan))
    
    # Combine Components
    adjusted_momentum_10d = momentum_10d * vol_adjustment
    adjusted_momentum_20d = momentum_20d * vol_adjustment
    
    # Take average of adjusted momentums
    factor = (adjusted_momentum_10d + adjusted_momentum_20d) / 2
    
    return factor
