import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factors based on momentum, volume dynamics, volatility regimes, and their interactions.
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    Series of composite alpha factor values
    """
    
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Basic price and volume calculations
    close = df['close']
    open_price = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Momentum Components
    ret_1d = close.pct_change(1)
    ret_3d = close.pct_change(3)
    ret_5d = close.pct_change(5)
    
    momentum_acceleration = ret_1d - ret_3d
    
    # Momentum persistence (count of consistent sign over 3 days)
    momentum_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(2, len(df)):
        if i >= 2:
            recent_rets = [ret_1d.iloc[i], ret_1d.iloc[i-1], ret_1d.iloc[i-2]]
            if len(recent_rets) == 3:
                signs = [1 if x > 0 else -1 if x < 0 else 0 for x in recent_rets]
                momentum_persistence.iloc[i] = sum(1 for j in range(1, 3) if signs[j] == signs[0])
    
    momentum_magnitude = abs(ret_1d) / (abs(ret_3d) + 0.001)
    
    # Price Behavior
    intraday_strength = (close - open_price) / (high - low + 0.001)
    price_efficiency = ret_1d / ((high - low) / close + 0.001)
    gap_behavior = (open_price / close.shift(1)) - 1
    
    # Volume Dynamics
    volume_trend_3d = volume.pct_change(3)
    volume_trend_5d = volume.pct_change(5)
    volume_acceleration = volume_trend_3d - volume_trend_5d
    
    volume_confirmation = np.sign(ret_1d) * np.sign(volume.pct_change(1))
    volume_intensity_return = ret_1d * (volume / volume.shift(1))
    volume_divergence = abs(ret_1d) * (1 - abs(volume_confirmation))
    
    # Volume persistence (count of increasing volume over 3 days)
    volume_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(2, len(df)):
        if i >= 2:
            recent_volumes = [volume.iloc[i] > volume.iloc[i-1], 
                            volume.iloc[i-1] > volume.iloc[i-2]]
            volume_persistence.iloc[i] = sum(recent_volumes)
    
    volume_stability = 1 / (abs(volume.pct_change(1)) + 0.001)
    
    # Volume breakout
    volume_breakout = pd.Series(index=df.index, dtype=bool)
    for i in range(3, len(df)):
        if i >= 3:
            volume_breakout.iloc[i] = volume.iloc[i] > max(volume.iloc[i-1], 
                                                          volume.iloc[i-2], 
                                                          volume.iloc[i-3])
    volume_breakout = volume_breakout.astype(float)
    
    # Volatility Regimes
    daily_range_ratio = (high - low) / close
    volatility_momentum = daily_range_ratio.pct_change(1)
    
    # Volatility persistence
    volatility_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(2, len(df)):
        if i >= 2:
            recent_ranges = [daily_range_ratio.iloc[i] > daily_range_ratio.iloc[i-1],
                           daily_range_ratio.iloc[i-1] > daily_range_ratio.iloc[i-2]]
            volatility_persistence.iloc[i] = sum(recent_ranges)
    
    # Volatility regimes
    rolling_5day_median_range = daily_range_ratio.rolling(5).median()
    high_vol_regime = (daily_range_ratio > rolling_5day_median_range).astype(float)
    low_vol_regime = (daily_range_ratio < rolling_5day_median_range).astype(float)
    volatility_expansion = (volatility_momentum > 0).astype(float)
    
    # Volatility-adjusted returns
    range_adjusted_return = ret_1d / (daily_range_ratio + 0.001)
    avg_3day_range = daily_range_ratio.rolling(3).mean()
    volatility_scaled_momentum = ret_3d / (avg_3day_range + 0.001)
    regime_adaptive_return = ret_1d * (1 + volatility_expansion)
    
    # Non-linear Interactions
    volume_weighted_momentum = ret_3d * (1 + volume_trend_3d)
    high_volume_acceleration = momentum_acceleration * volume_breakout
    stable_volume_momentum = ret_3d * volume_stability
    
    low_vol_quality_momentum = ret_1d * low_vol_regime
    high_vol_momentum_strength = momentum_magnitude * high_vol_regime
    volatility_adaptive_acceleration = momentum_acceleration * volatility_expansion
    
    short_term_volume_momentum = ret_1d * volume_trend_3d
    medium_term_volatility_momentum = ret_3d * volatility_momentum
    persistent_regime_momentum = ret_5d * (volume_persistence / 3) * (volatility_persistence / 3)
    
    # Robust Composite Factors
    # Volume-Confirmed Momentum
    volume_confirmed_momentum = ret_3d * (1 + volume_trend_3d)
    
    # Volatility-Adjusted Daily Return
    volatility_adjusted_daily_return = ret_1d / (daily_range_ratio + 0.001)
    
    # Regime-Enhanced Momentum
    regime_enhanced_momentum = ret_5d * (volume_persistence / 3) * (1 + volatility_persistence / 3)
    
    # Final Alpha Signals - Equal weighted combination
    vwsm = volume_confirmed_momentum  # Volume-Weighted Short Momentum
    vadr = volatility_adjusted_daily_return  # Volatility-Adjusted Daily Return
    mrpm = regime_enhanced_momentum  # Multi-Regime Persistent Momentum
    
    # Composite alpha factor (equal weighted combination)
    alpha_factor = (vwsm.fillna(0) + vadr.fillna(0) + mrpm.fillna(0)) / 3
    
    return alpha_factor
