import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate momentum components with exponential decay
    close = df['close']
    volume = df['volume']
    
    # Short-term momentum (5-day)
    short_momentum = (close / close.shift(5)) - 1
    short_weights = np.exp(-0.2 * np.arange(len(short_momentum)))
    short_weighted = short_momentum.rolling(window=len(short_momentum), min_periods=6).apply(
        lambda x: np.sum(x * short_weights[:len(x)]) / np.sum(short_weights[:len(x)]), raw=True
    )
    
    # Medium-term momentum (20-day)
    medium_momentum = (close / close.shift(20)) - 1
    medium_weights = np.exp(-0.05 * np.arange(len(medium_momentum)))
    medium_weighted = medium_momentum.rolling(window=len(medium_momentum), min_periods=21).apply(
        lambda x: np.sum(x * medium_weights[:len(x)]) / np.sum(medium_weights[:len(x)]), raw=True
    )
    
    # Volume confirmation signals
    volume_5d_ratio = volume / volume.shift(5)
    volume_20d_ratio = volume / volume.shift(20)
    
    # Volume-weighted momentum components
    short_volume_weighted = short_weighted * volume_5d_ratio
    medium_volume_weighted = medium_weighted * volume_20d_ratio
    
    # Combined momentum with volume confirmation
    combined_momentum = 0.6 * short_volume_weighted + 0.4 * medium_volume_weighted
    
    # Volatility scaling using high-low range
    daily_range = (df['high'] - df['low']) / df['close']
    volatility_20d = daily_range.rolling(window=20, min_periods=10).std()
    volatility_scaled_momentum = combined_momentum / (volatility_20d + 1e-8)
    
    # Divergence penalty and convergence reward
    momentum_sign = np.sign(volatility_scaled_momentum)
    volume_sign_5d = np.sign(volume_5d_ratio - 1)
    volume_sign_20d = np.sign(volume_20d_ratio - 1)
    
    # Calculate divergence/convergence signals
    divergence_5d = momentum_sign != volume_sign_5d
    divergence_20d = momentum_sign != volume_sign_20d
    convergence_5d = momentum_sign == volume_sign_5d
    convergence_20d = momentum_sign == volume_sign_20d
    
    # Apply penalties and rewards
    penalty = -0.1 * (divergence_5d.astype(float) + divergence_20d.astype(float))
    reward = 0.1 * (convergence_5d.astype(float) + convergence_20d.astype(float))
    
    # Final factor calculation
    final_factor = volatility_scaled_momentum + penalty + reward
    
    return final_factor
