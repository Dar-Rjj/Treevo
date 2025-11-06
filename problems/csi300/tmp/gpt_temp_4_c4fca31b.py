import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Weighted Price Acceleration Divergence factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Compute first derivative (velocity) using t-1 and t-3
    velocity = (df['close'].shift(1) - df['close'].shift(3)) / 2
    
    # Compute second derivative (acceleration) using velocity at t-1 and t-4
    acceleration = (velocity.shift(1) - velocity.shift(4)) / 3
    
    # Identify acceleration regime vs 15-day rolling median
    acceleration_median = acceleration.rolling(window=15, min_periods=10).median()
    acceleration_regime = np.where(acceleration > acceleration_median, 1, -1)
    
    # Calculate volume momentum
    volume_ema_8 = df['volume'].ewm(span=8, adjust=False).mean()
    volume_momentum = (df['volume'].shift(1) - volume_ema_8.shift(1)) / volume_ema_8.shift(1)
    
    # Compute volume-price divergence using Spearman correlation
    def spearman_corr(x):
        if len(x) < 5:
            return np.nan
        price_changes = x['close'].pct_change().dropna()
        volume_changes = x['volume'].pct_change().dropna()
        if len(price_changes) < 5 or len(volume_changes) < 5:
            return np.nan
        min_len = min(len(price_changes), len(volume_changes))
        return pd.Series(price_changes.iloc[:min_len]).corr(pd.Series(volume_changes.iloc[:min_len]), method='spearman')
    
    # Calculate rolling correlation
    divergence_corr = pd.Series(index=df.index, dtype=float)
    for i in range(12, len(df)):
        window_data = df.iloc[i-12:i]
        divergence_corr.iloc[i] = spearman_corr(window_data)
    
    # Create divergence magnitude factor
    divergence_factor = 1 - np.abs(divergence_corr)
    
    # Combine acceleration and volume momentum
    accel_volume_component = acceleration * volume_momentum
    
    # Adjust for divergence signal
    divergence_adjusted = accel_volume_component * divergence_factor
    
    # Incorporate bid-ask spread proxy using high-low range
    daily_range = (df['high'] - df['low']) / df['close']
    spread_ratio = daily_range.rolling(window=5, min_periods=3).mean()
    
    # Scale factor by spread-to-price ratio (inverse relationship)
    spread_scaled = divergence_adjusted / (1 + spread_ratio)
    
    # Apply regime-dependent transformation
    def piecewise_transform(x, regime):
        if regime == 1:  # High acceleration regime
            return np.tanh(x * 0.5)
        else:  # Low acceleration regime
            return np.sign(x) * np.sqrt(np.abs(x))
    
    # Apply transformation based on acceleration regime
    transformed_factor = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if not np.isnan(spread_scaled.iloc[i]) and not np.isnan(acceleration_regime[i]):
            transformed_factor.iloc[i] = piecewise_transform(spread_scaled.iloc[i], acceleration_regime[i])
        else:
            transformed_factor.iloc[i] = np.nan
    
    # Add market state memory component
    # Track recent factor performance in different regimes
    regime_performance = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        recent_data = transformed_factor.iloc[i-20:i]
        recent_regimes = acceleration_regime[i-20:i]
        
        if len(recent_data.dropna()) > 5:
            # Calculate regime-specific performance (absolute returns correlation proxy)
            high_accel_perf = recent_data[recent_regimes == 1].mean() if np.sum(recent_regimes == 1) > 2 else 0
            low_accel_perf = recent_data[recent_regimes == -1].mean() if np.sum(recent_regimes == -1) > 2 else 0
            
            # Adjust weight based on recent efficacy
            current_regime = acceleration_regime[i]
            if current_regime == 1 and high_accel_perf > 0:
                regime_weight = 1.2
            elif current_regime == -1 and low_accel_perf > 0:
                regime_weight = 1.2
            else:
                regime_weight = 0.8
                
            regime_performance.iloc[i] = regime_weight
        else:
            regime_performance.iloc[i] = 1.0
    
    # Final factor with regime weighting
    final_factor = transformed_factor * regime_performance
    
    return final_factor
