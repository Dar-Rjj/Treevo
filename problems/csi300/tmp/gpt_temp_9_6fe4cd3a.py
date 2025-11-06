import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe volatility-adjusted momentum with volume divergence detection and regime-aware scaling.
    
    Interpretation:
    - Combines short-term (2-day) and medium-term (5-day) momentum signals
    - Detects volume divergences where price and volume movements contradict
    - Scales by true range to normalize for volatility across different stocks
    - Uses regime detection to adjust weights based on market conditions
    - Positive values indicate strong momentum with volume confirmation in trending markets
    - Negative values suggest weak momentum or distribution patterns
    """
    
    # Multi-timeframe momentum adjusted by volatility
    momentum_2d = (df['close'] - df['close'].shift(2)) / (df['high'] - df['low']).rolling(window=2).mean().replace(0, 1e-7)
    momentum_5d = (df['close'] - df['close'].shift(5)) / (df['high'] - df['low']).rolling(window=5).mean().replace(0, 1e-7)
    
    # Volume divergence detection
    volume_trend = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    price_trend = df['close'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_divergence = np.sign(price_trend) * np.sign(volume_trend) * np.abs(volume_trend)
    
    # True range for volatility scaling
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Regime detection (trending vs mean-reverting)
    price_volatility = df['close'].pct_change().rolling(window=10).std()
    regime_strength = price_volatility.rolling(window=5).std() / (price_volatility + 1e-7)
    
    # Combine components with regime-aware weights
    trending_weight = np.where(regime_strength > regime_strength.median(), 0.7, 0.3)
    mean_reverting_weight = 1 - trending_weight
    
    alpha_factor = (
        trending_weight * (0.4 * momentum_2d + 0.3 * momentum_5d) +
        mean_reverting_weight * (0.2 * momentum_2d + 0.1 * momentum_5d) +
        0.2 * volume_divergence / (true_range + 1e-7)
    )
    
    return alpha_factor
