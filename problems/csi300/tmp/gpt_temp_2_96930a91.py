import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Acceleration
    # Short-Term Momentum (3-day)
    short_term_momentum = (data['close'] / data['close'].shift(3) - 1)
    
    # Medium-Term Momentum (10-day)
    medium_term_momentum = (data['close'] / data['close'].shift(10) - 1)
    
    # Momentum Acceleration
    momentum_acceleration = (short_term_momentum - medium_term_momentum) / (abs(medium_term_momentum) + 1e-8)
    
    # Volume Divergence Component
    # Calculate returns and volume changes
    returns_5d = data['close'].pct_change(periods=5)
    volume_changes = data['volume'].pct_change(periods=5)
    
    # Price-Volume Correlation (5-day rolling)
    price_volume_corr = data['close'].pct_change().rolling(window=5).corr(data['volume'].pct_change())
    
    # Volume Trend Strength
    # Calculate volume slope over 8 days
    def linear_slope(x):
        if len(x) < 2:
            return np.nan
        return np.polyfit(range(len(x)), x, 1)[0]
    
    volume_slope = data['volume'].rolling(window=8).apply(linear_slope, raw=False)
    price_slope = data['close'].rolling(window=8).apply(linear_slope, raw=False)
    
    # Volume trend strength relative to price
    volume_trend_strength = volume_slope / (abs(price_slope) + 1e-8)
    
    # Identify Divergence Pattern
    # Detect when volume trend opposes price momentum
    divergence_detected = (momentum_acceleration > 0) & (volume_trend_strength < 0) | (momentum_acceleration < 0) & (volume_trend_strength > 0)
    
    # Quantify divergence strength using residual analysis
    # Residual from price-volume relationship
    price_volume_residual = returns_5d - (price_volume_corr * volume_changes)
    divergence_strength = abs(price_volume_residual) * divergence_detected.astype(float)
    
    # Signal Synthesis
    # Combine Acceleration and Divergence
    raw_signal = momentum_acceleration * divergence_strength
    
    # Apply exponential weighting with 5-day half-life
    alpha = 1 - np.exp(np.log(0.5) / 5)  # Decay factor for 5-day half-life
    weighted_signal = raw_signal.ewm(alpha=alpha, adjust=False).mean()
    
    # Regime-Based Filtering
    # Calculate rolling volatility (20-day)
    volatility = data['close'].pct_change().rolling(window=20).std()
    
    # Create volatility quintiles
    vol_quintiles = volatility.rolling(window=60, min_periods=20).apply(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop').iloc[-1] if len(x.dropna()) >= 20 else np.nan, 
        raw=False
    )
    
    # Regime persistence (how long in current quintile)
    regime_persistence = vol_quintiles.groupby(vol_quintiles).transform(
        lambda x: x.expanding().count()
    )
    
    # Adjust signal strength based on regime
    # Higher signals in stable regimes (middle quintiles with high persistence)
    regime_filter = np.where(
        (vol_quintiles == 2) | (vol_quintiles == 3),
        np.minimum(1.0, regime_persistence / 10),  # Boost in stable regimes
        np.maximum(0.3, 1.0 - regime_persistence / 20)  # Reduce in extreme regimes
    )
    
    final_signal = weighted_signal * regime_filter
    
    return final_signal
