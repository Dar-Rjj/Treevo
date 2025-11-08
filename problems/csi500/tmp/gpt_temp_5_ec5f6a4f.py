import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Volume Convergence Index alpha factor
    Combines price momentum and volume momentum across multiple timeframes
    to identify strong convergence patterns for predicting future returns.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Price Momentum Component
    # Multiple timeframe momentums
    mom_5 = df['close'].pct_change(5)
    mom_10 = df['close'].pct_change(10)
    mom_20 = df['close'].pct_change(20)
    
    # Momentum acceleration (rate of change of momentum)
    mom_accel_5_10 = mom_5 - mom_10.shift(5)
    mom_accel_10_20 = mom_10 - mom_20.shift(10)
    
    # Normalize price momentum components
    price_momentum = (mom_5.rolling(window=20, min_periods=10).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() != 0 else 0) +
                     mom_10.rolling(window=20, min_periods=10).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() != 0 else 0) +
                     mom_accel_5_10.rolling(window=20, min_periods=10).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() != 0 else 0) +
                     mom_accel_10_20.rolling(window=20, min_periods=10).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() != 0 else 0)) / 4
    
    # Volume Momentum Component
    # Volume trend strength
    vol_ma_5 = df['volume'].rolling(window=5, min_periods=3).mean()
    vol_ma_20 = df['volume'].rolling(window=20, min_periods=10).mean()
    vol_trend = (vol_ma_5 / vol_ma_20 - 1)
    
    # Volume acceleration
    vol_mom_5 = df['volume'].pct_change(5)
    vol_mom_10 = df['volume'].pct_change(10)
    vol_accel = vol_mom_5 - vol_mom_10.shift(5)
    
    # Normalize volume momentum components
    volume_momentum = (vol_trend.rolling(window=20, min_periods=10).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() != 0 else 0) +
                      vol_accel.rolling(window=20, min_periods=10).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() != 0 else 0)) / 2
    
    # Convergence Scoring
    # Strong convergence: both price and volume momentum accelerating in same direction
    convergence_direction = np.sign(price_momentum) * np.sign(volume_momentum)
    convergence_strength = (abs(price_momentum) + abs(volume_momentum)) / 2
    
    # Weight by convergence persistence (how long the convergence has been maintained)
    convergence_persistence = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if convergence_direction.iloc[i] == convergence_direction.iloc[i-1] and convergence_direction.iloc[i] != 0:
            convergence_persistence.iloc[i] = convergence_persistence.iloc[i-1] + 1
    
    # Final alpha factor: convergence score weighted by persistence
    alpha_factor = convergence_direction * convergence_strength * (1 + np.log1p(convergence_persistence) / 10)
    
    # Handle NaN values
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
