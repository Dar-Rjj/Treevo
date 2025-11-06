import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Fibonacci-Weighted Geometric Convergence Factor with True Range Scaling
    
    Economic intuition: Combines Fibonacci time horizons (3, 5, 8, 13, 21 days) with geometric mean 
    smoothing to capture multiplicative convergence patterns. The factor identifies stocks where 
    price momentum, volume intensity, and volatility compression align across Fibonacci sequences, 
    normalized by true range for volatility scaling and weighted by dollar volume for liquidity adjustment.
    
    Key innovations:
    - Fibonacci sequence horizons (3, 5, 8, 13, 21) for natural market rhythm alignment
    - Individual geometric smoothing of components before combination
    - True range normalization for proper volatility scaling
    - Dollar volume weighting for liquidity-adjusted signals
    - Multiplicative convergence detection across multiple timeframes
    """
    
    # Fibonacci horizons
    fib_horizons = [3, 5, 8, 13, 21]
    
    # Calculate true range
    true_range = pd.DataFrame({
        'high_low': df['high'] - df['low'],
        'high_close_prev': abs(df['high'] - df['close'].shift(1)),
        'low_close_prev': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    
    # Initialize component arrays
    momentum_components = []
    volume_components = []
    volatility_components = []
    
    for horizon in fib_horizons:
        # Momentum component with geometric smoothing
        momentum_raw = (df['close'] / df['close'].shift(horizon) - 1)
        momentum_smooth = momentum_raw.rolling(window=horizon).apply(
            lambda x: (x + 1).prod() ** (1/len(x)) - 1 if all(x.notna()) else np.nan
        )
        momentum_components.append(momentum_smooth)
        
        # Volume intensity with geometric smoothing
        volume_intensity_raw = df['volume'] / df['volume'].rolling(window=horizon).mean()
        volume_smooth = volume_intensity_raw.rolling(window=horizon).apply(
            lambda x: x.prod() ** (1/len(x)) if all(x.notna()) else np.nan
        )
        volume_components.append(volume_smooth)
        
        # Volatility compression with geometric smoothing
        daily_vol = (df['high'] - df['low']) / df['close']
        vol_compression_raw = daily_vol / daily_vol.rolling(window=horizon).mean()
        vol_smooth = vol_compression_raw.rolling(window=horizon).apply(
            lambda x: x.prod() ** (1/len(x)) if all(x.notna()) else np.nan
        )
        volatility_components.append(vol_smooth)
    
    # Geometric mean across Fibonacci horizons for each component
    momentum_geo = pd.concat(momentum_components, axis=1).apply(
        lambda x: x.prod() ** (1/len(x)) if all(x.notna()) else np.nan, axis=1
    )
    volume_geo = pd.concat(volume_components, axis=1).apply(
        lambda x: x.prod() ** (1/len(x)) if all(x.notna()) else np.nan, axis=1
    )
    volatility_geo = pd.concat(volatility_components, axis=1).apply(
        lambda x: x.prod() ** (1/len(x)) if all(x.notna()) else np.nan, axis=1
    )
    
    # Core multiplicative convergence factor
    convergence_factor = momentum_geo * volume_geo / volatility_geo
    
    # Normalize by true range for volatility scaling
    avg_true_range = true_range.rolling(window=21).mean()
    range_normalized = convergence_factor / avg_true_range
    
    # Weight by dollar volume for liquidity adjustment
    dollar_volume = df['close'] * df['volume']
    dollar_weight = dollar_volume / dollar_volume.rolling(window=21).mean()
    
    # Final factor with true range scaling and dollar volume weighting
    factor = range_normalized * dollar_weight
    
    return factor
