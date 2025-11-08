import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Acceleration Component
    df = df.copy()
    
    # First momentum: Close(t) - Close(t-1)
    momentum = df['close'].diff(1)
    
    # Second-order momentum (acceleration): Momentum(t) - Momentum(t-1)
    acceleration = momentum.diff(1)
    
    # Acceleration magnitude and direction
    accel_magnitude = np.abs(acceleration)
    accel_direction = np.sign(acceleration)
    
    # Efficiency Momentum Component
    # 5-day absolute price change
    abs_price_change_5d = df['close'].diff(1).abs().rolling(window=5, min_periods=3).sum()
    
    # 5-day true range sum
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_sum_5d = true_range.rolling(window=5, min_periods=3).sum()
    
    # Efficiency ratio
    efficiency_ratio = abs_price_change_5d / tr_sum_5d
    efficiency_ratio = efficiency_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Efficiency momentum
    efficiency_momentum = efficiency_ratio.diff(1)
    efficiency_direction = np.sign(efficiency_momentum)
    
    # Acceleration-Efficiency Divergence
    # Identify divergence patterns
    bullish_divergence = (accel_direction < 0) & (efficiency_direction > 0)
    bearish_divergence = (accel_direction > 0) & (efficiency_direction < 0)
    
    # Calculate divergence strength
    divergence_strength = acceleration * efficiency_momentum
    
    # Assess divergence persistence over 3-day window
    divergence_persistence = pd.Series(0, index=df.index)
    for i in range(2, len(df)):
        if i >= 2:
            window_divergence = divergence_strength.iloc[i-2:i+1]
            persistence_score = (window_divergence > 0).sum() if bullish_divergence.iloc[i] else (window_divergence < 0).sum() if bearish_divergence.iloc[i] else 0
            divergence_persistence.iloc[i] = persistence_score / 3.0
    
    # Divergence quality score
    divergence_quality = divergence_strength * divergence_persistence
    
    # Volume Integration
    # Volume breakout strength
    volume_ma_20 = df['volume'].rolling(window=20, min_periods=10).mean()
    volume_ratio = df['volume'] / volume_ma_20
    
    # Volume trend consistency over 5 days
    volume_trend = df['volume'].rolling(window=5, min_periods=3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    volume_trend_strength = np.abs(volume_trend) / df['volume'].rolling(window=5, min_periods=3).std()
    volume_trend_strength = volume_trend_strength.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Combine with divergence signal
    volume_adjusted_divergence = divergence_quality * volume_ratio
    
    # Volume direction confirmation
    volume_confirmation = np.sign(volume_trend) * np.sign(divergence_quality)
    volume_confirmed_signal = volume_adjusted_divergence * (1 + 0.5 * (volume_confirmation > 0))
    
    # Volume persistence enhancement
    volume_persistence = (df['volume'] > volume_ma_20).rolling(window=5, min_periods=3).mean()
    final_volume_signal = volume_confirmed_signal * (1 + volume_persistence)
    
    # Volatility Context Enhancement
    # 20-day price volatility
    price_volatility = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    
    # Flag high volatility periods (above 80th percentile)
    volatility_threshold = price_volatility.rolling(window=100, min_periods=50).quantile(0.8)
    high_vol_regime = price_volatility > volatility_threshold
    
    # Apply regime weighting
    volatility_weight = 1 + (high_vol_regime * price_volatility / price_volatility.rolling(window=100, min_periods=50).mean())
    
    # Generate final composite factor
    final_factor = final_volume_signal * volatility_weight
    
    # Clean and return
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return final_factor
