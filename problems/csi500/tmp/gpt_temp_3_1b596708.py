import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volume-Price Momentum with Adaptive Regimes alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Extract price and volume data
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Price Momentum Analysis
    # Multi-horizon returns
    ret_3d = close / close.shift(3) - 1
    ret_10d = close / close.shift(10) - 1
    ret_21d = close / close.shift(21) - 1
    
    # Momentum acceleration
    ret_5d = close / close.shift(5) - 1
    ret_15d = close / close.shift(15) - 1
    accel_short = ret_3d - ret_5d
    accel_medium = ret_10d - ret_15d
    
    # Volume Leadership Signals
    # Volume momentum across horizons
    vol_mom_3d = volume / volume.shift(3) - 1
    vol_mom_10d = volume / volume.shift(10) - 1
    vol_mom_21d = volume / volume.shift(21) - 1
    
    # Volume-price timing relationships
    vol_lead_corr = pd.Series(index=df.index, dtype=float)
    vol_conf_corr = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        if i >= 3:
            # Volume leading price correlation (volume[t-3:t-1] vs returns[t])
            vol_window = volume.iloc[i-3:i]
            ret_window = (close.iloc[i] / close.iloc[i-1] - 1)
            vol_lead_corr.iloc[i] = 0  # Simplified for single point correlation
            
            # Volume confirmation correlation (volume[t-1:t] vs returns[t])
            vol_conf_window = volume.iloc[i-1:i+1]
            if len(vol_conf_window) > 1:
                vol_conf_corr.iloc[i] = np.corrcoef(vol_conf_window.values, 
                                                   [ret_window, ret_window])[0,1]
    
    # Volume divergence
    vol_div_short = np.sign(vol_mom_3d) * np.sign(ret_3d)
    vol_div_medium = np.sign(vol_mom_10d) * np.sign(ret_10d)
    
    # Adaptive Threshold Framework
    # Rolling percentiles
    window = 20
    
    # Price momentum thresholds
    ret_3d_30p = ret_3d.rolling(window).quantile(0.3)
    ret_3d_70p = ret_3d.rolling(window).quantile(0.7)
    
    # Volume momentum thresholds
    vol_mom_40p = vol_mom_3d.rolling(window).quantile(0.4)
    vol_mom_60p = vol_mom_3d.rolling(window).quantile(0.6)
    
    # Range-based volatility
    daily_range = (high - low) / close
    range_30p = daily_range.rolling(window).quantile(0.3)
    range_70p = daily_range.rolling(window).quantile(0.7)
    
    # Volume surge threshold
    vol_80p = volume.rolling(window).quantile(0.8)
    
    # Regime detection
    high_vol_regime = daily_range > range_70p
    low_vol_regime = daily_range < range_30p
    volume_surge = volume > vol_80p
    
    # Multi-Timeframe Correlation Analysis
    # Cross-horizon momentum consistency
    corr_short_medium = pd.Series(index=df.index, dtype=float)
    corr_medium_long = pd.Series(index=df.index, dtype=float)
    
    for i in range(25, len(df)):
        if i >= 10:
            # Short-medium correlation (5-day window)
            short_window = ret_3d.iloc[i-5:i]
            medium_window = ret_10d.iloc[i-5:i]
            if len(short_window) > 1 and len(medium_window) > 1:
                corr_short_medium.iloc[i] = np.corrcoef(short_window.values, medium_window.values)[0,1]
            
            # Medium-long correlation (10-day window)
            medium_window2 = ret_10d.iloc[i-10:i]
            long_window = ret_21d.iloc[i-10:i]
            if len(medium_window2) > 1 and len(long_window) > 1:
                corr_medium_long.iloc[i] = np.corrcoef(medium_window2.values, long_window.values)[0,1]
    
    all_horizon_alignment = (corr_short_medium + corr_medium_long) / 2
    
    # Volume-momentum synchronization
    vol_price_sync_short = pd.Series(index=df.index, dtype=float)
    vol_price_sync_medium = pd.Series(index=df.index, dtype=float)
    
    for i in range(25, len(df)):
        if i >= 10:
            # Short-term sync (5-day window)
            price_short = ret_3d.iloc[i-5:i]
            vol_short = vol_mom_3d.iloc[i-5:i]
            if len(price_short) > 1 and len(vol_short) > 1:
                vol_price_sync_short.iloc[i] = np.corrcoef(price_short.values, vol_short.values)[0,1]
            
            # Medium-term sync (10-day window)
            price_medium = ret_10d.iloc[i-10:i]
            vol_medium = vol_mom_10d.iloc[i-10:i]
            if len(price_medium) > 1 and len(vol_medium) > 1:
                vol_price_sync_medium.iloc[i] = np.corrcoef(price_medium.values, vol_medium.values)[0,1]
    
    sync_persistence = (vol_price_sync_short + vol_price_sync_medium) / 2
    
    # Intelligent Signal Combination
    # Momentum strength scoring
    momentum_consistency = (np.sign(ret_3d) == np.sign(ret_10d)) & (np.sign(ret_10d) == np.sign(ret_21d))
    momentum_strength = (ret_3d.abs() + ret_10d.abs() + ret_21d.abs()) / 3
    acceleration_bonus = (accel_short + accel_medium) / 2
    cross_timeframe_weight = all_horizon_alignment.fillna(0)
    
    momentum_score = (momentum_strength * (1 + 0.2 * momentum_consistency.astype(float)) + 
                     0.3 * acceleration_bonus) * (1 + cross_timeframe_weight)
    
    # Volume confirmation scoring
    volume_leadership = vol_lead_corr.fillna(0)
    volume_synchronization = sync_persistence.fillna(0)
    volume_divergence_penalty = ((vol_div_short < 0) | (vol_div_medium < 0)).astype(float) * -0.5
    
    volume_score = (1 + 0.4 * volume_leadership + 
                   0.3 * volume_synchronization + 
                   volume_divergence_penalty)
    
    # Regime-adaptive weighting
    regime_weight = pd.Series(1.0, index=df.index)
    regime_weight[high_vol_regime] = 1.3  # Increased weight on volume leadership
    regime_weight[low_vol_regime] = 0.8   # Decreased weight on volume, increased on momentum
    regime_weight[volume_surge] = 1.2     # Increased weight on volume-price correlation
    
    # Final Alpha Output
    alpha = momentum_score * volume_score * regime_weight
    
    # Handle NaN values
    alpha = alpha.fillna(0)
    
    return alpha
