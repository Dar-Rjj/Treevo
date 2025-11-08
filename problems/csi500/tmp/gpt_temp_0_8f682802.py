import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Momentum Acceleration with Volume-Price Divergence alpha factor
    """
    # Extract price and volume data
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Multi-Timeframe Momentum Components
    mom_3d = close / close.shift(3) - 1
    mom_8d = close / close.shift(8) - 1
    mom_21d = close / close.shift(21) - 1
    
    # Momentum Acceleration Calculation
    accel_short = mom_3d - mom_8d
    accel_medium = mom_8d - mom_21d
    accel_composite = (accel_short + accel_medium) / 2
    
    # Volume-Price Divergence Analysis
    vol_mom_5d = volume / volume.shift(5) - 1
    vol_mom_10d = volume / volume.shift(10) - 1
    
    # Price-volume alignment
    price_vol_alignment = np.where((mom_3d > 0) & (vol_mom_5d > 0), 1,
                          np.where((mom_3d < 0) & (vol_mom_5d < 0), -1, 0))
    
    # Divergence strength
    divergence_strength = np.abs(mom_3d) - np.abs(vol_mom_5d)
    
    # Exponential Smoothing Application
    # Momentum smoothing
    smooth_mom_3d = pd.Series(index=close.index, dtype=float)
    smooth_mom_8d = pd.Series(index=close.index, dtype=float)
    
    for i in range(len(close)):
        if i == 0:
            smooth_mom_3d.iloc[i] = mom_3d.iloc[i]
            smooth_mom_8d.iloc[i] = mom_8d.iloc[i]
        else:
            smooth_mom_3d.iloc[i] = 0.7 * mom_3d.iloc[i] + 0.3 * smooth_mom_3d.iloc[i-1]
            smooth_mom_8d.iloc[i] = 0.85 * mom_8d.iloc[i] + 0.15 * smooth_mom_8d.iloc[i-1]
    
    # Volume smoothing
    smooth_vol_mom = pd.Series(index=close.index, dtype=float)
    for i in range(len(close)):
        if i == 0:
            smooth_vol_mom.iloc[i] = vol_mom_5d.iloc[i]
        else:
            smooth_vol_mom.iloc[i] = 0.6 * vol_mom_5d.iloc[i] + 0.4 * smooth_vol_mom.iloc[i-1]
    
    # Amount-based regime indicator
    amount_trend = amount / amount.shift(15) - 1
    smooth_amount = pd.Series(index=close.index, dtype=float)
    for i in range(len(close)):
        if i == 0:
            smooth_amount.iloc[i] = amount_trend.iloc[i]
        else:
            smooth_amount.iloc[i] = 0.75 * amount_trend.iloc[i] + 0.25 * smooth_amount.iloc[i-1]
    
    # Volatility context
    daily_range = (high - low) / close
    avg_range_20d = daily_range.rolling(window=20).mean()
    high_vol_regime = daily_range > avg_range_20d
    
    # Regime-Based Signal Weighting
    high_participation = smooth_amount > 0
    
    # Base signal calculation
    base_signal = pd.Series(index=close.index, dtype=float)
    
    for i in range(len(close)):
        if high_participation.iloc[i]:
            # High participation regime
            vol_weight = 0.7
            mom_weight = 0.3
            vol_confirmation = 1.5 if price_vol_alignment[i] != 0 else 1.0
        else:
            # Low participation regime
            vol_weight = 0.3
            mom_weight = 0.7
            vol_confirmation = 1.0
        
        # Volatility scaling
        if high_vol_regime.iloc[i]:
            vol_scale = 1.0 / (daily_range.iloc[i] + 1e-6)
        else:
            vol_scale = 1.2  # Amplify in low volatility
        
        # Calculate base signal
        momentum_component = (smooth_mom_3d.iloc[i] * 0.6 + smooth_mom_8d.iloc[i] * 0.4)
        volume_component = smooth_vol_mom.iloc[i] * divergence_strength.iloc[i]
        
        base_signal.iloc[i] = (momentum_component * mom_weight + 
                              volume_component * vol_weight) * vol_confirmation * vol_scale
    
    # Apply final smoothing and normalization
    final_factor = base_signal.rolling(window=5).mean()
    
    return final_factor
