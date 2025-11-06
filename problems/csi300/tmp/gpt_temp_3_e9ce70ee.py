import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Volatility-normalized momentum with volume confirmation across aligned timeframes.
    Detects regimes via volatility clustering and directional volume shifts.
    """
    
    # Volatility clustering detection - 20-day rolling volatility
    returns = df['close'].pct_change()
    volatility_20d = returns.rolling(window=20, min_periods=10).std()
    volatility_5d = returns.rolling(window=5, min_periods=3).std()
    
    # Volatility regime indicator (high vol vs low vol)
    vol_regime = volatility_5d / (volatility_20d + 1e-7)
    
    # Multi-timeframe momentum (5-day and 10-day) normalized by volatility
    momentum_5d = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) * volatility_20d + 1e-7)
    momentum_10d = (df['close'] - df['close'].shift(10)) / (df['close'].shift(10) * volatility_20d + 1e-7)
    
    # Aligned timeframe volume confirmation
    volume_5d_avg = df['volume'].rolling(window=5, min_periods=3).mean()
    volume_10d_avg = df['volume'].rolling(window=10, min_periods=5).mean()
    
    volume_confirm_5d = df['volume'] / volume_5d_avg
    volume_confirm_10d = df['volume'] / volume_10d_avg
    
    # Directional volume shift - volume trend aligned with price direction
    volume_trend = df['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.corrcoef(x, range(len(x)))[0,1] if len(x) > 1 and np.std(x) > 0 else 0
    )
    price_trend = df['close'].rolling(window=5, min_periods=3).apply(
        lambda x: np.corrcoef(x, range(len(x)))[0,1] if len(x) > 1 and np.std(x) > 0 else 0
    )
    directional_volume = np.sign(volume_trend * price_trend) * np.abs(volume_trend)
    
    # Multiplicative combination with regime adjustment
    alpha_factor = (
        momentum_5d * volume_confirm_5d * 
        momentum_10d * volume_confirm_10d * 
        directional_volume * vol_regime
    )
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
