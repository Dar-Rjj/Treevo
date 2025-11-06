import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Fractal Momentum Divergence factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # 1. Multi-Scale Regime Identification
    # Volatility Fractal Analysis
    returns = data['close'].pct_change()
    vol_5d = returns.rolling(window=5).std()
    vol_20d = returns.rolling(window=20).std()
    vol_ratio = vol_5d / vol_20d
    
    # Fractal dimension estimation using Hurst exponent approximation
    def estimate_fractal_dimension(series, window=20):
        lags = [1, 2, 5, 10]
        hurst_vals = []
        for lag in lags:
            if len(series) > window:
                ts = series.rolling(window=window).mean().dropna()
                if len(ts) > max(lags):
                    lagged = ts.shift(lag)
                    diff = (ts - lagged).dropna()
                    if len(diff) > 0:
                        rs = diff.rolling(window=window-lag).std()
                        if not rs.isna().all():
                            hurst = np.log(rs.mean() + 1e-10) / np.log(lag)
                            hurst_vals.append(hurst)
        if hurst_vals:
            fractal_dim = 2 - np.mean(hurst_vals)
            return max(0.5, min(2.0, fractal_dim))
        return 1.0
    
    # Calculate volatility fractal dimension
    vol_fractal = vol_20d.rolling(window=50).apply(
        lambda x: estimate_fractal_dimension(x) if len(x.dropna()) >= 20 else 1.0, 
        raw=False
    )
    
    # Volatility regime classification
    vol_regime = pd.Series(1, index=data.index)  # Normal by default
    vol_regime[vol_fractal > 1.5] = 2  # High Fractal
    vol_regime[vol_fractal < 0.67] = 0  # Low Fractal
    
    # Range Microstructure Analysis
    daily_range = (data['high'] - data['low']) / data['close'].shift(1)
    range_5d_avg = daily_range.rolling(window=5).mean()
    range_20d_avg = daily_range.rolling(window=20).mean()
    range_ratio = range_5d_avg / range_20d_avg
    
    # Microstructure regime classification
    micro_regime = pd.Series(1, index=data.index)  # Normal by default
    micro_regime[range_ratio < 0.7] = 0  # Compressed
    micro_regime[range_ratio > 1.3] = 2  # Expanded
    
    # 2. Fractal Momentum Analysis
    # Multi-Scale Price Acceleration
    returns_3d = returns.rolling(window=3).mean()  # Velocity
    acceleration = returns_3d.diff().rolling(window=3).mean()  # Acceleration
    
    # Volume-Weighted Fractal Momentum
    vwap = (data['close'] * data['volume']).rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    vwap_velocity = vwap.pct_change().rolling(window=3).mean()
    vwap_acceleration = vwap_velocity.diff().rolling(window=3).mean()
    
    # Volume-price fractal correlation
    volume_returns_corr = data['volume'].rolling(window=10).corr(returns.abs())
    
    # 3. Smart Money Fractal Integration
    # Large transaction analysis
    avg_amount = data['amount'].rolling(window=20).mean()
    large_txn_threshold = avg_amount * 2
    large_txn_ratio = (data['amount'] > large_txn_threshold).rolling(window=5).mean()
    
    # Price impact efficiency
    price_impact = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-10)
    txn_efficiency = (large_txn_ratio * price_impact.abs()).rolling(window=5).mean()
    
    # 4. Regime-Specific Fractal Signals
    fractal_signals = pd.Series(0.0, index=data.index)
    
    # High Fractal Volatility regime
    high_vol_mask = vol_regime == 2
    if high_vol_mask.any():
        # Multi-scale acceleration divergence
        acc_5d = acceleration.rolling(window=5).mean()
        acc_10d = acceleration.rolling(window=10).mean()
        fractal_signals[high_vol_mask] = (acc_5d - acc_10d)[high_vol_mask] * vol_fractal[high_vol_mask]
    
    # Normal + Compressed regime
    normal_compressed_mask = (vol_regime == 1) & (micro_regime == 0)
    if normal_compressed_mask.any():
        # Fractal dimension × acceleration persistence
        acc_persistence = acceleration.rolling(window=5).std()
        fractal_signals[normal_compressed_mask] = (vol_fractal * acc_persistence * volume_returns_corr)[normal_compressed_mask]
    
    # Low Fractal + Expanded regime
    low_expanded_mask = (vol_regime == 0) & (micro_regime == 2)
    if low_expanded_mask.any():
        # Geometric breakout detection
        breakout_strength = (daily_range * vwap_acceleration.abs())[low_expanded_mask]
        fractal_signals[low_expanded_mask] = breakout_strength * txn_efficiency[low_expanded_mask]
    
    # Default regime (catch-all)
    default_mask = fractal_signals == 0
    if default_mask.any():
        fractal_signals[default_mask] = (acceleration * vwap_velocity * volume_returns_corr)[default_mask]
    
    # 5. Composite Fractal Factor Generation
    # Volume fractal confirmation
    volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
    volume_multiplier = np.tanh(volume_ratio - 1)  # Scale between -1 and 1
    
    # Microstructure adjustment
    micro_adjustment = 1.0 + (micro_regime - 1) * 0.3  # Compressed: 0.7, Normal: 1.0, Expanded: 1.3
    
    # Entanglement signals (simplified)
    price_volume_entanglement = (returns.abs() * data['volume'].pct_change().abs()).rolling(window=5).mean()
    
    # Final composite factor
    composite_factor = (
        fractal_signals * 
        volume_multiplier * 
        micro_adjustment * 
        (1 + price_volume_entanglement)
    )
    
    # Fractal signal enhancement
    # Signed power transformation
    enhanced_factor = np.sign(composite_factor) * (np.abs(composite_factor) ** 0.7)
    
    # Scale by current fractal volatility
    vol_scaling = 1.0 / (vol_5d + 1e-10)
    final_factor = enhanced_factor * vol_scaling
    
    # Topological breakout detection adjustment
    breakout_adjustment = (daily_range > daily_range.rolling(window=20).quantile(0.8)).astype(float)
    final_factor = final_factor * (1 + 0.5 * breakout_adjustment)
    
    return final_factor.fillna(0)
