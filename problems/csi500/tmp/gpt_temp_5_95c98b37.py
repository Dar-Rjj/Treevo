import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Clean momentum-acceleration with volume confirmation and robust volatility normalization
    # Uses longer timeframes for more stable signals and cleaner factor construction
    
    # Momentum acceleration over longer timeframe (10-day momentum rate of change)
    momentum_10d = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    momentum_acceleration = momentum_10d - momentum_10d.shift(5)
    
    # Volume confirmation using rolling percentiles (cleaner than min/max)
    volume_rank = df['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.2)) / (x.quantile(0.8) - x.quantile(0.2) + 1e-7),
        raw=False
    )
    
    # Robust volatility normalization using modified ATR with longer window
    true_range = np.maximum(
        np.maximum(
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1))
        ),
        abs(df['low'] - df['close'].shift(1))
    )
    volatility_normalizer = true_range.rolling(window=20).mean()
    
    # Clean interaction: momentum acceleration conditioned on volume confirmation
    raw_factor = momentum_acceleration * np.clip(volume_rank, -2, 2)
    
    # Final factor with robust volatility normalization
    factor = raw_factor / (volatility_normalizer + 1e-7)
    
    return factor
