import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Extract price and volume data
    close = df['close']
    volume = df['volume']
    
    # Momentum Components
    # Short-Term Momentum (5-day)
    mom_5d = (close / close.shift(5)) - 1
    mom_accel = (close / close.shift(5)) - (close.shift(1) / close.shift(6))
    
    # Medium-Term Momentum (10-day)
    mom_10d = (close / close.shift(10)) - 1
    mom_curvature = (close / close.shift(10)) - 2*(close.shift(5) / close.shift(15)) + (close.shift(10) / close.shift(20))
    
    # Momentum Regime Detection
    mom_convergence = ((mom_5d > 0) & (mom_10d > 0)).astype(int) - ((mom_5d < 0) & (mom_10d < 0)).astype(int)
    
    # Volatility Components
    # Short-Term Volatility (5-day)
    returns_5d = close.pct_change().rolling(window=5).std()
    
    # Medium-Term Volatility (10-day)
    returns_10d = close.pct_change().rolling(window=10).std()
    
    # Volatility Regime Detection
    vol_ratio = returns_5d / returns_10d
    vol_expansion = (vol_ratio > 1.2).astype(int)
    vol_contraction = (vol_ratio < 0.8).astype(int)
    
    # Volume Components
    # Volume Trend Analysis
    volume_5d_avg = volume.rolling(window=5).mean()
    volume_ratio = volume / volume_5d_avg
    volume_mom = (volume / volume.shift(5)) - 1
    volume_10d_avg = volume.rolling(window=10).mean()
    volume_breakout = (volume > volume_10d_avg * 1.5).astype(int)
    
    # Volume Volatility Patterns
    volume_cv = volume.rolling(window=5).std() / volume.rolling(window=5).mean()
    volume_stability = 1 / (1 + volume_cv)
    volume_spike = (volume > volume.rolling(window=20).mean() + 2 * volume.rolling(window=20).std()).astype(int)
    
    # Multi-Timeframe Volume Confirmation
    volume_trend_5d = volume.rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_trend_10d = volume.rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_regime_shift = ((volume_trend_5d > 0) & (volume_trend_10d < 0)).astype(int) - ((volume_trend_5d < 0) & (volume_trend_10d > 0)).astype(int)
    
    # Signal Integration & Non-linear Interactions
    # Volatility-Adjusted Momentum
    vol_adj_mom_5d = mom_5d / (returns_5d + 1e-8)
    vol_adj_mom_10d = mom_10d / (returns_10d + 1e-8)
    vol_change = returns_5d / returns_10d.shift(5)
    mom_accel_vol_adj = mom_accel / (vol_change + 1e-8)
    
    # Volume-Confirmed Signals
    vol_conf_mom_5d = vol_adj_mom_5d * volume_ratio
    vol_conf_mom_10d = vol_adj_mom_10d * volume_ratio
    stability_adj_mom_5d = vol_conf_mom_5d * volume_stability
    stability_adj_mom_10d = vol_conf_mom_10d * volume_stability
    breakout_enhanced_mom = (stability_adj_mom_5d + stability_adj_mom_10d) * (1 + 0.5 * volume_breakout)
    
    # Regime-Based Weighting
    convergence_weight = 1 + 0.3 * mom_convergence
    volatility_weight = 1 - 0.4 * vol_expansion + 0.2 * vol_contraction
    volume_conf_weight = 1 + 0.25 * ((volume_ratio > 1.2) & (volume_mom > 0)).astype(int)
    
    # Composite Factor Construction
    weighted_mom_signals = (
        convergence_weight * volatility_weight * volume_conf_weight * 
        (0.4 * breakout_enhanced_mom + 0.3 * mom_accel_vol_adj + 0.3 * mom_curvature)
    )
    
    # Volume-based filters
    volume_filter = (volume_ratio > 0.7) & (volume_stability > 0.3)
    filtered_signal = weighted_mom_signals * volume_filter
    
    # Incorporate regime shift detection
    regime_enhanced = filtered_signal * (1 + 0.2 * volume_regime_shift)
    
    # Final alpha factor with smoothing
    alpha_factor = regime_enhanced.rolling(window=3).mean()
    
    return alpha_factor
