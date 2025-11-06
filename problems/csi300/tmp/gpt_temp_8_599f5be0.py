import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-Horizon Momentum-Volume Convergence with Volatility Regime Adjustment
    
    Economic intuition: Captures the consistency of price momentum and volume expansion 
    across multiple time horizons using geometric means, while adjusting for volatility 
    regimes. The factor identifies stocks where bullish signals are persistent across 
    different market cycles and supported by volume confirmation in favorable volatility 
    environments.
    
    Key innovations:
    - Triple-horizon geometric alignment (3, 5, 8 days) for momentum and volume
    - Volatility regime classification using rolling percentiles
    - Multiplicative combination with regime-dependent weighting
    - Clean normalization using rolling standard deviation
    - Focus on cross-horizon signal convergence
    """
    
    # Momentum components across multiple horizons
    momentum_3d = df['close'] / df['close'].shift(3) - 1
    momentum_5d = df['close'] / df['close'].shift(5) - 1
    momentum_8d = df['close'] / df['close'].shift(8) - 1
    
    # Volume intensity ratios across matching horizons
    volume_intensity_3d = df['volume'] / df['volume'].rolling(window=3).mean()
    volume_intensity_5d = df['volume'] / df['volume'].rolling(window=5).mean()
    volume_intensity_8d = df['volume'] / df['volume'].rolling(window=8).mean()
    
    # Volatility components using true range
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    
    vol_3d = true_range.rolling(window=3).mean() / df['close']
    vol_5d = true_range.rolling(window=5).mean() / df['close']
    vol_8d = true_range.rolling(window=8).mean() / df['close']
    
    # Geometric mean alignment across horizons
    momentum_geo = (momentum_3d * momentum_5d * momentum_8d) ** (1/3)
    volume_geo = (volume_intensity_3d * volume_intensity_5d * volume_intensity_8d) ** (1/3)
    volatility_geo = (vol_3d * vol_5d * vol_8d) ** (1/3)
    
    # Volatility regime classification
    vol_regime = volatility_geo.rolling(window=21).apply(
        lambda x: 1 if x.iloc[-1] <= x.quantile(0.33) else (0.5 if x.iloc[-1] <= x.quantile(0.67) else 0.25)
    )
    
    # Core factor with regime-dependent weighting
    raw_factor = momentum_geo * volume_geo * vol_regime / volatility_geo
    
    # Clean volatility normalization
    factor_volatility = raw_factor.rolling(window=21).std()
    factor = raw_factor / factor_volatility
    
    return factor
