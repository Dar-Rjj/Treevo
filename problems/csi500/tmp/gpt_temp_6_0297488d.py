import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Dynamic High-Low Volatility-Adjusted Momentum factor
    
    Combines momentum signal with intraday volatility adjustment and volume confirmation
    """
    # Calculate 5-day momentum using close prices
    momentum = data['close'].pct_change(periods=5)
    
    # Calculate intraday volatility as (High - Low) / Close
    intraday_volatility = (data['high'] - data['low']) / data['close']
    
    # Adjust momentum by volatility (avoid division by zero)
    volatility_adjusted_momentum = momentum / (intraday_volatility + 1e-8)
    
    # Calculate 10-day volume rank percentile
    volume_rank = data['volume'].rolling(window=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], 
        raw=False
    )
    
    # Apply volume confirmation by multiplying with volume rank
    factor = volatility_adjusted_momentum * volume_rank
    
    return factor
