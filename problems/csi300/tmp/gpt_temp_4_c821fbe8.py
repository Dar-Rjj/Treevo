import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum
    short_term_return = df['close'].pct_change(periods=5)
    medium_term_return = df['close'].pct_change(periods=21)
    
    # Compute Volume Dynamics
    daily_volume_changes = df['volume'].pct_change()
    
    # Calculate rolling correlation between price returns and lagged volume
    daily_returns = df['close'].pct_change()
    volume_correlations = pd.Series(index=df.index, dtype=float)
    
    for i in range(10, len(df)):
        start_idx = i - 9
        end_idx = i
        price_window = daily_returns.iloc[start_idx:end_idx+1]
        volume_window = df['volume'].iloc[start_idx-1:end_idx]  # Lagged volume
        if len(price_window) == len(volume_window) and len(price_window) >= 3:
            correlation = price_window.corr(volume_window)
            volume_correlations.iloc[i] = correlation if not pd.isna(correlation) else 0
    
    # Calculate Dynamic Volatility Adjustment
    daily_range = (df['high'] - df['low']) / df['close']
    range_volatility = daily_range.rolling(window=10, min_periods=5).std()
    return_volatility = daily_returns.rolling(window=10, min_periods=5).std()
    
    combined_volatility = range_volatility * return_volatility
    volatility_smoothed = combined_volatility.rolling(window=5, min_periods=3).mean()
    
    # Generate Final Alpha Factor
    # Compute Divergence Strength
    correlation_avg = volume_correlations.rolling(window=20, min_periods=10).mean()
    divergence_strength = (volume_correlations - correlation_avg).abs()
    volatility_weighted_divergence = divergence_strength * volatility_smoothed
    
    # Apply Directional and Volume Confirmation
    trend_direction = np.sign(short_term_return)
    volume_to_price_ratio = df['volume'] / df['close']
    
    # Final factor calculation
    factor = volatility_weighted_divergence * trend_direction * volume_to_price_ratio
    
    return factor.fillna(0)
