import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Multi-Horizon Reversal with Volume Confirmation
    """
    close = df['close']
    volume = df['volume']
    
    # Multi-Horizon Price Reversal components
    reversal_2d = (close.shift(2) / close - 1) * volume
    reversal_5d = (close.shift(5) / close - 1) * volume
    reversal_10d = (close.shift(10) / close - 1) * volume
    reversal_20d = (close.shift(20) / close - 1) * volume
    
    # Dynamic Volatility Regime Classification
    returns = close.pct_change()
    
    vol_3d = returns.rolling(window=3).std()
    vol_7d = returns.rolling(window=7).std()
    vol_15d = returns.rolling(window=15).std()
    
    vol_ratio_3_15 = vol_3d / vol_15d
    vol_ratio_7_15 = vol_7d / vol_15d
    
    # Volatility regime classification
    high_vol_regime = vol_ratio_3_15 > 1.5
    low_vol_regime = vol_ratio_7_15 < 0.6
    normal_vol_regime = ~(high_vol_regime | low_vol_regime)
    
    # Volume Confirmation Framework
    # Volume Momentum Hierarchy
    vol_momentum_2d = volume / volume.shift(2) - 1
    vol_momentum_5d = volume / volume.shift(5) - 1
    vol_momentum_10d = volume / volume.shift(10) - 1
    
    # Volume-Price Synchronization
    price_change_2d = close / close.shift(2) - 1
    price_change_5d = close / close.shift(5) - 1
    price_change_10d = close / close.shift(10) - 1
    
    sync_2d = np.sign(price_change_2d) * np.sign(vol_momentum_2d)
    sync_5d = np.sign(price_change_5d) * np.sign(vol_momentum_5d)
    sync_10d = np.sign(price_change_10d) * np.sign(vol_momentum_10d)
    
    # Volume Regime Power Scaling
    vol_sma_15 = volume.rolling(window=15).mean()
    
    high_volume = volume > (2.0 * vol_sma_15)
    low_volume = volume < (0.5 * vol_sma_15)
    normal_volume = ~(high_volume | low_volume)
    
    # Nonlinear Volume Transformation
    volume_power = pd.Series(index=volume.index, dtype=float)
    volume_power[high_volume] = volume[high_volume] ** 0.5
    volume_power[low_volume] = volume[low_volume] ** 1.5
    volume_power[normal_volume] = volume[normal_volume] ** 1.0
    
    # Adaptive Alpha Integration
    alpha = pd.Series(index=close.index, dtype=float)
    
    # Regime-Matched Reversal Selection with Volume Synchronization and Intensity Scaling
    for idx in close.index:
        if high_vol_regime.loc[idx]:
            base_reversal = reversal_2d.loc[idx]
            sync_multiplier = sync_2d.loc[idx]
        elif low_vol_regime.loc[idx]:
            base_reversal = reversal_20d.loc[idx]
            sync_multiplier = sync_10d.loc[idx]  # Use medium-term sync for long-term reversal
        else:  # normal volatility
            base_reversal = reversal_10d.loc[idx]
            sync_multiplier = sync_10d.loc[idx]
        
        # Apply volume synchronization multiplier
        synchronized_reversal = base_reversal * (1 + 0.5 * sync_multiplier)
        
        # Apply volume intensity scaling
        if high_volume.loc[idx]:
            final_alpha = synchronized_reversal * volume_power.loc[idx] * 1.2
        elif low_volume.loc[idx]:
            final_alpha = synchronized_reversal * volume_power.loc[idx] * 0.8
        else:
            final_alpha = synchronized_reversal * volume_power.loc[idx]
        
        alpha.loc[idx] = final_alpha
    
    return alpha
