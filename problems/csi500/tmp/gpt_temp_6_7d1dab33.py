import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Asymmetric Volatility-Adjusted Price Momentum factor
    Combines momentum with volatility asymmetry and volume confirmation
    """
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    # Calculate Price Momentum
    # Short-term momentum (5-day)
    mom_short = close / close.shift(5) - 1
    
    # Medium-term momentum (20-day)
    mom_medium = close / close.shift(20) - 1
    
    # Momentum ratio (relative performance)
    momentum_ratio = (mom_short / mom_medium) - 1
    
    # Calculate Asymmetric Volatility
    # Daily true range
    true_range = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    
    # Identify up and down days
    up_days = close > close.shift(1)
    down_days = close < close.shift(1)
    
    # Calculate rolling volatility measures (20-day window)
    upside_vol = true_range.rolling(window=20).apply(
        lambda x: x[up_days.reindex(x.index)].mean(), raw=False
    )
    downside_vol = true_range.rolling(window=20).apply(
        lambda x: x[down_days.reindex(x.index)].mean(), raw=False
    )
    
    # Handle division by zero and calculate volatility asymmetry
    volatility_asymmetry = np.log(upside_vol / downside_vol.replace(0, np.nan))
    
    # Apply Volume Confirmation
    # Volume trend ratio (5-day vs 20-day average)
    vol_short = volume.rolling(window=5).mean()
    vol_medium = volume.rolling(window=20).mean()
    volume_trend = vol_short / vol_medium
    
    # Combine all components
    # Multiply momentum ratio by volatility asymmetry and volume trend
    alpha_factor = momentum_ratio * volatility_asymmetry * volume_trend
    
    return alpha_factor
