import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Enhanced multi-timeframe momentum with regime-aware volume divergence and volatility scaling
    # Economic intuition: Momentum persistence is strongest when confirmed by volume divergence across timeframes
    # and scaled by relative volatility conditions for regime adaptation
    
    # Multi-timeframe momentum with exponential smoothing for robustness
    momentum_5d = (df['close'] / df['close'].shift(5) - 1).ewm(span=3).mean()
    momentum_10d = (df['close'] / df['close'].shift(10) - 1).ewm(span=5).mean()
    momentum_20d = (df['close'] / df['close'].shift(20) - 1).ewm(span=8).mean()
    
    # Momentum convergence score - measures alignment across timeframes
    momentum_signs = np.sign(momentum_5d) + np.sign(momentum_10d) + np.sign(momentum_20d)
    momentum_magnitude = (abs(momentum_5d) * abs(momentum_10d) * abs(momentum_20d)) ** (1/3)
    momentum_convergence = (momentum_signs / 3) * momentum_magnitude
    
    # Volume divergence factor - compares current volume to multiple historical windows
    volume_short_ratio = df['volume'] / (df['volume'].rolling(window=5).mean() + 1e-7)
    volume_medium_ratio = df['volume'] / (df['volume'].rolling(window=15).mean() + 1e-7)
    volume_long_ratio = df['volume'] / (df['volume'].rolling(window=30).mean() + 1e-7)
    
    # Volume divergence score - emphasizes consistent volume patterns
    volume_divergence = (
        np.tanh(volume_short_ratio - 1) + 
        np.tanh(volume_medium_ratio - 1) + 
        np.tanh(volume_long_ratio - 1)
    ) / 3
    
    # Regime-aware volatility scaling using multiple volatility measures
    range_volatility = (df['high'] - df['low']).rolling(window=10).std()
    close_volatility = df['close'].pct_change().rolling(window=10).std()
    
    # Combined volatility measure with regime detection
    combined_volatility = (range_volatility / df['close'] + close_volatility) / 2
    volatility_regime = combined_volatility.rolling(window=20).apply(
        lambda x: 1 if x.iloc[-1] > x.quantile(0.7) else 
                  -1 if x.iloc[-1] < x.quantile(0.3) else 0
    )
    
    # Adaptive volatility scaling based on regime
    base_vol_scale = 1 / (combined_volatility + 1e-7)
    regime_adjusted_scale = base_vol_scale * (1 + 0.5 * volatility_regime)
    
    # Final factor: momentum convergence amplified by volume divergence and regime-adjusted volatility
    factor = momentum_convergence * (1 + volume_divergence) * regime_adjusted_scale
    
    return factor
