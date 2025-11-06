import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Novel factor: Volatility-adjusted momentum with volume persistence
    # Combines price momentum, volume trend persistence, and volatility normalization
    # Interpretable as: Stocks with sustained momentum supported by persistent volume activity in stable volatility regimes tend to continue trending
    
    # Price momentum components
    short_momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)  # 3-day momentum
    medium_momentum = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)  # 8-day momentum
    
    # Momentum persistence (agreement between short and medium term)
    momentum_persistence = np.sign(short_momentum) * np.sign(medium_momentum) * (abs(short_momentum) + abs(medium_momentum))
    
    # Volume persistence components
    current_volume_ratio = df['volume'] / df['volume'].rolling(window=5).mean()
    volume_trend_strength = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(5), x, 1)[0]) / df['volume'].rolling(window=5).mean()
    
    # Volume persistence score (combining level and trend)
    volume_persistence = current_volume_ratio * (1 + volume_trend_strength)
    
    # Volatility regime assessment
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    
    volatility_regime = true_range.rolling(window=5).mean() / df['close'].rolling(window=5).mean()
    volatility_stability = 1 / (volatility_regime + 1e-7)
    
    # Combine all components
    # Higher scores indicate persistent momentum with volume confirmation in stable low-volatility environments
    factor = momentum_persistence * volume_persistence * volatility_stability
    
    return factor
