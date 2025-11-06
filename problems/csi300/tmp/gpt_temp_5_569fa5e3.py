import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Price Momentum
    close = df['close']
    price_momentum_1d = close.pct_change(1)
    price_momentum_3d = close.pct_change(3)
    price_momentum_5d = close.pct_change(5)
    
    # Combine price momentum using weighted average
    price_momentum = (0.5 * price_momentum_1d + 
                      0.3 * price_momentum_3d + 
                      0.2 * price_momentum_5d)
    
    # Compute Volume Momentum
    volume = df['volume']
    volume_roc_1d = volume.pct_change(1)
    volume_roc_3d = volume.pct_change(3)
    volume_roc_5d = volume.pct_change(5)
    
    # Combine volume momentum using weighted average
    volume_momentum = (0.5 * volume_roc_1d + 
                       0.3 * volume_roc_3d + 
                       0.2 * volume_roc_5d)
    
    # Calculate Price-Volume Divergence
    divergence = (price_momentum - volume_momentum) * np.sign(price_momentum)
    
    # Identify Volatility Regime
    returns = close.pct_change()
    vol_20d = returns.rolling(window=20, min_periods=10).std()
    vol_regime = vol_20d.rolling(window=60, min_periods=30).apply(
        lambda x: 1 if x.iloc[-1] > x.median() else 0, raw=False
    )
    
    # Adjust Divergence Signal by Regime
    # Amplify in low volatility (regime=0), dampen in high volatility (regime=1)
    regime_adjustment = np.where(vol_regime == 0, 1.5, 0.7)
    adjusted_divergence = divergence * regime_adjustment
    
    # Normalize the final factor
    factor = (adjusted_divergence - adjusted_divergence.rolling(window=60, min_periods=30).mean()) / \
             adjusted_divergence.rolling(window=60, min_periods=30).std()
    
    return factor
