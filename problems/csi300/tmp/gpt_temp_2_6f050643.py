import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Scaled Reversal with Volume-Price Divergence alpha factor
    """
    close = df['close']
    volume = df['volume']
    
    # Volatility-Scaled Short-Term Reversal
    # 3-day reversal
    ret_3d = close.pct_change(3) * -1
    
    # 5-day reversal
    ret_5d = close.pct_change(5) * -1
    
    # 5-day volatility (standard deviation of returns)
    vol_5d = close.pct_change().rolling(window=5).std()
    
    # Scale reversals by volatility
    scaled_rev_3d = ret_3d / (vol_5d + 1e-8)
    scaled_rev_5d = ret_5d / (vol_5d + 1e-8)
    
    # Volume-Price Divergence Detection
    # Price momentum
    price_mom_5d = close.pct_change(5)
    price_mom_10d = close.pct_change(10)
    
    # Volume momentum
    vol_mom_5d = volume.pct_change(5)
    vol_mom_10d = volume.pct_change(10)
    
    # Divergence signal generation
    divergence_5d = np.where(
        (price_mom_5d * vol_mom_5d) < 0,
        (price_mom_5d - vol_mom_5d) * np.sign(price_mom_5d),
        0
    )
    
    divergence_10d = np.where(
        (price_mom_10d * vol_mom_10d) < 0,
        (price_mom_10d - vol_mom_10d) * np.sign(price_mom_10d),
        0
    )
    
    # Dynamic Regime-Aware Thresholding
    # Recent volatility regime
    vol_20d = close.pct_change().rolling(window=20).std()
    vol_regime = vol_20d / vol_20d.rolling(window=60).median()
    
    # Regime-based weighting
    reversal_weight = np.where(vol_regime > 1.2, 1.5, 1.0)
    divergence_weight = np.where(vol_regime < 0.8, 1.3, 1.0)
    
    # Volume Confirmation Mechanism
    # Volume trends
    vol_trend_5d = volume.rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    vol_trend_10d = volume.rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Volume-signal alignment
    vol_conf_reversal = np.where(
        (scaled_rev_3d * vol_trend_5d) > 0,
        1.2,
        np.where((scaled_rev_3d * vol_trend_5d) < 0, 0.8, 1.0)
    )
    
    vol_conf_divergence = np.where(
        (divergence_5d * vol_trend_10d) > 0,
        1.1,
        np.where((divergence_5d * vol_trend_10d) < 0, 0.9, 1.0)
    )
    
    # Composite Alpha Construction
    # Combine scaled components with regime-aware weights
    composite_reversal = (scaled_rev_3d + scaled_rev_5d) * reversal_weight
    composite_divergence = (divergence_5d + divergence_10d) * divergence_weight
    
    # Apply volume confirmation
    alpha = (composite_reversal * vol_conf_reversal + 
             composite_divergence * vol_conf_divergence)
    
    return pd.Series(alpha, index=df.index)
