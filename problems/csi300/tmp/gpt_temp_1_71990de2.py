import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe volatility-normalized momentum with volume trend confirmation
    # Combines short-term and medium-term momentum, normalized by volatility, with volume-based confirmation
    
    # Short-term momentum (5-day) and medium-term momentum (10-day)
    mom_short = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    mom_medium = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Volatility normalization using multiple timeframes
    vol_short = (df['high'] - df['low']).rolling(window=5).std()
    vol_medium = (df['high'] - df['low']).rolling(window=10).std()
    
    # Volume trend confirmation with multiple perspectives
    volume_ratio = df['volume'] / df['volume'].rolling(window=10).mean()
    volume_accel = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)
    
    # Clean momentum signal: short-term momentum normalized by short-term volatility
    clean_mom = mom_short / (vol_short + 1e-7)
    
    # Momentum consistency: alignment between short and medium-term momentum
    mom_alignment = np.sign(mom_short) * np.sign(mom_medium) * (abs(mom_short) + abs(mom_medium))
    
    # Volume extreme reversal: detect potential reversal points at volume extremes
    volume_percentile = df['volume'].rolling(window=20).apply(lambda x: (x[-1] - x.mean()) / x.std())
    volume_reversal = -np.sign(mom_short) * abs(volume_percentile) * (abs(volume_percentile) > 1.5)
    
    # Core factor: synergistic combination
    # Combines clean momentum, momentum alignment, and volume-based signals
    factor = (clean_mom * mom_alignment * volume_ratio) + (volume_reversal * volume_accel)
    
    return factor
