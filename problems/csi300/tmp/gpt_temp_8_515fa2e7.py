import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Divergence with Momentum Acceleration factor
    """
    # Calculate price and volume returns for various timeframes
    df = df.copy()
    
    # Price and volume calculations
    df['price_return_1d'] = df['close'].pct_change()
    df['volume_return_1d'] = df['volume'].pct_change()
    
    # Multi-timeframe divergence ratios
    # Short-term (3-day): price acceleration / volume acceleration
    df['price_accel_3d'] = df['price_return_1d'].rolling(window=3).mean() - df['price_return_1d'].rolling(window=6).mean()
    df['volume_accel_3d'] = df['volume_return_1d'].rolling(window=3).mean() - df['volume_return_1d'].rolling(window=6).mean()
    df['divergence_short'] = df['price_accel_3d'] / (df['volume_accel_3d'] + 1e-8)
    
    # Medium-term (8-day): price momentum slope / volume momentum slope
    df['price_momentum_8d'] = df['close'].rolling(window=8).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    df['volume_momentum_8d'] = df['volume'].rolling(window=8).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    df['divergence_medium'] = df['price_momentum_8d'] / (df['volume_momentum_8d'] + 1e-8)
    
    # Long-term (15-day): price volatility trend / volume volatility trend
    df['price_vol_trend'] = df['close'].pct_change().rolling(window=5).std() - df['close'].pct_change().rolling(window=15).std()
    df['volume_vol_trend'] = df['volume'].pct_change().rolling(window=5).std() - df['volume'].pct_change().rolling(window=15).std()
    df['divergence_long'] = df['price_vol_trend'] / (df['volume_vol_trend'] + 1e-8)
    
    # Assess Momentum Quality
    # Acceleration persistence (consecutive same-sign price accelerations)
    df['price_accel_sign'] = np.sign(df['price_accel_3d'])
    df['persistence_count'] = 0
    for i in range(1, len(df)):
        if df['price_accel_sign'].iloc[i] == df['price_accel_sign'].iloc[i-1]:
            df['persistence_count'].iloc[i] = df['persistence_count'].iloc[i-1] + 1
    
    # Momentum strength (sum of 5-day accelerations / sum of absolute accelerations)
    df['momentum_strength'] = (
        df['price_accel_3d'].rolling(window=5).sum() / 
        (df['price_accel_3d'].abs().rolling(window=5).sum() + 1e-8)
    )
    
    # Evaluate Volume Momentum Stability
    # Volume acceleration trend (7-day slope)
    df['volume_accel_trend'] = df['volume_accel_3d'].rolling(window=7).apply(
        lambda x: (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 0
    )
    
    # Volume momentum volatility (std of 5-day accelerations / mean of absolute accelerations)
    df['volume_momentum_vol'] = (
        df['volume_accel_3d'].rolling(window=5).std() / 
        (df['volume_accel_3d'].abs().rolling(window=5).mean() + 1e-8)
    )
    
    # Volume momentum stability score (inverse of volatility)
    df['volume_stability'] = 1 / (df['volume_momentum_vol'] + 1e-8)
    
    # Detect Reversal Signals
    # Divergence extremes
    df['divergence_extreme'] = (
        (df['divergence_short'] > 4.0) | (df['divergence_short'] < 0.25) |
        (df['divergence_medium'] > 4.0) | (df['divergence_medium'] < 0.25) |
        (df['divergence_long'] > 4.0) | (df['divergence_long'] < 0.25)
    ).astype(int)
    
    # Momentum fatigue (current persistence / max historical persistence)
    df['max_persistence'] = df['persistence_count'].rolling(window=20, min_periods=1).max()
    df['momentum_fatigue'] = df['persistence_count'] / (df['max_persistence'] + 1e-8)
    
    # Generate Alpha Factor
    # Combine divergence ratios with momentum quality
    divergence_composite = (
        0.4 * df['divergence_short'] + 
        0.35 * df['divergence_medium'] + 
        0.25 * df['divergence_long']
    ) * df['momentum_strength']
    
    # Weight by volume momentum stability
    stability_weighted = divergence_composite * df['volume_stability']
    
    # Filter using reversal signals (reduce factor when divergence extremes detected)
    filtered_factor = stability_weighted * (1 - 0.5 * df['divergence_extreme'])
    
    # Scale by momentum fatigue indicators (reduce factor when momentum is fatigued)
    final_factor = filtered_factor * (1 - df['momentum_fatigue'])
    
    # Clean and return
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    final_factor = final_factor.fillna(0)
    
    return final_factor
