import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Quantum Microstructure Momentum Convergence factor
    Combines quantum price dynamics, microstructure patterns, volume analysis, 
    cross-asset convergence, and efficiency detection in a regime-adaptive framework
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Quantum Price Dynamics
    price_echo = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    wave_interference = ((df['high'] - df['close']) * (df['close'] - df['low'])) / \
                       ((df['high'] - df['low'] + 1e-8) ** 2)
    
    entanglement = ((df['close'] - df['close'].shift(1)) * \
                   (df['volume'] - df['volume'].shift(1))) / \
                   (df['high'] - df['low'] + 1e-8)
    
    # Microstructure Patterns
    tunneling = np.abs(df['close'] - (df['high'] + df['low']) / 2) / \
                (df['high'] - df['low'] + 1e-8)
    
    level_states = np.where(
        ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8) > 0.7) | 
        ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8) > 0.7),
        1.5, 1.0
    )
    
    momentum_leap = (df['close'] / df['close'].shift(2) - 1) / \
                   (df['close'] / df['close'].shift(4) - 1 + 1e-8)
    
    # Volume Microstructure
    volume_state = df['volume'] / df['volume'].rolling(window=3, min_periods=1).min()
    
    # Volume persistence (consecutive days below rolling average)
    volume_ma = df['volume'].rolling(window=4, min_periods=1).mean()
    volume_persistence = (df['volume'] < volume_ma.shift(1)).rolling(window=5, min_periods=1).sum()
    
    microstructure_field = price_echo * (df['volume'] / (df['volume'].shift(1) + 1e-8))
    
    # Efficiency Detection
    volume_momentum = df['volume'] / df['volume'].rolling(window=4, min_periods=1).mean().shift(1)
    
    range_efficiency = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    price_rejection = (np.minimum(df['close'], df['open']) - df['low']) - \
                     (df['high'] - np.maximum(df['close'], df['open']))
    
    # Cross-Asset Convergence (using rolling percentiles as proxy for peer comparison)
    micro_momentum = price_echo.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if len(x) >= 10 else np.nan
    )
    
    meso_momentum = momentum_leap.rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 10 else np.nan
    )
    
    macro_momentum = entanglement.rolling(window=20, min_periods=10).apply(
        lambda x: (x > 0).sum() / len(x) if len(x) >= 10 else np.nan
    )
    
    # Regime Detection
    efficiency_regime = range_efficiency.rolling(window=10, min_periods=5).mean()
    high_efficiency = efficiency_regime > efficiency_regime.rolling(window=20, min_periods=10).quantile(0.7)
    low_efficiency = efficiency_regime < efficiency_regime.rolling(window=20, min_periods=10).quantile(0.3)
    
    # Regime-Adaptive Synthesis
    high_eff_component = price_echo * micro_momentum * (1 - volume_momentum.clip(0, 2) / 2)
    low_eff_component = volume_persistence * (micro_momentum + meso_momentum + macro_momentum) / 3
    compression_component = momentum_leap * price_rejection
    expansion_component = microstructure_field * level_states
    
    # Final Alpha Combination
    alpha = (
        high_eff_component * high_efficiency.astype(float) +
        low_eff_component * low_efficiency.astype(float) +
        compression_component * (1 - efficiency_regime) +
        expansion_component * efficiency_regime
    )
    
    # Normalize and clean
    alpha_clean = alpha.replace([np.inf, -np.inf], np.nan)
    alpha_zscore = (alpha_clean - alpha_clean.rolling(window=20, min_periods=10).mean()) / \
                   (alpha_clean.rolling(window=20, min_periods=10).std() + 1e-8)
    
    result = alpha_zscore.fillna(0)
    
    return result
