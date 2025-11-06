import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Calculate 20-day rolling volatility
    vol_20d = returns.rolling(window=20).std()
    
    # Calculate 252-day rolling volatility percentile
    vol_percentile = vol_20d.rolling(window=252).apply(lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 60)) if len(x.dropna()) > 0 else np.nan, raw=False)
    
    # Identify high volatility periods (above 60th percentile)
    high_vol_regime = (vol_percentile > 0.5).astype(int)
    
    # Calculate price and volume changes for different regimes
    price_change_3d = df['close'].pct_change(3)
    volume_change_3d = df['volume'].pct_change(3)
    
    price_change_10d = df['close'].pct_change(10)
    volume_change_10d = df['volume'].pct_change(10)
    
    # Calculate rolling correlations
    corr_3d = price_change_3d.rolling(window=20).corr(volume_change_3d)
    corr_10d = price_change_10d.rolling(window=20).corr(volume_change_10d)
    
    # Generate regime-specific signals
    high_vol_signal = -corr_3d  # Inverse correlation for high volatility
    normal_vol_signal = corr_10d  # Direct correlation for normal volatility
    
    # Apply asymmetric weighting based on volatility persistence
    vol_persistence = high_vol_regime.rolling(window=5).mean()
    high_vol_weight = 1.5 + 0.5 * vol_persistence  # Amplify during high volatility
    normal_vol_weight = 1.0 - 0.2 * vol_persistence  # Reduce during high volatility persistence
    
    # Combine signals using regime-conditional weights
    factor = (
        high_vol_regime * high_vol_weight * high_vol_signal + 
        (1 - high_vol_regime) * normal_vol_weight * normal_vol_signal
    )
    
    return factor
