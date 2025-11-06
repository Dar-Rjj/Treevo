import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-regime momentum synergy with volume-amount divergence and volatility-adaptive smoothing
    # Economic intuition: Combines momentum convergence across timeframes with volume-amount confirmation
    # and volatility regime filtering for robust trend quality assessment
    
    # Multi-timeframe momentum convergence
    momentum_5 = df['close'] / df['close'].shift(5) - 1
    momentum_10 = df['close'] / df['close'].shift(10) - 1
    momentum_20 = df['close'] / df['close'].shift(20) - 1
    
    # Momentum synergy score - geometric mean with sign alignment
    momentum_synergy = (
        np.sign(momentum_5) * np.sign(momentum_10) * np.sign(momentum_20) *
        (abs(momentum_5) * abs(momentum_10) * abs(momentum_20)) ** (1/3)
    )
    
    # Volume-amount divergence: discrepancy between volume and dollar amount trends
    volume_trend = df['volume'] / df['volume'].rolling(window=15).mean() - 1
    amount_trend = df['amount'] / df['amount'].rolling(window=15).mean() - 1
    volume_amount_divergence = np.arctan(volume_trend - amount_trend)  # Smooth bounded divergence measure
    
    # Volatility regime filter using adaptive rolling percentiles
    true_range = np.maximum(
        np.maximum(
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1))
        ),
        abs(df['low'] - df['close'].shift(1))
    )
    vol_regime = true_range.rolling(window=20).apply(
        lambda x: np.percentile(x, 70) if len(x) == 20 else np.nan
    )
    volatility_filter = 1 / (1 + np.exp((true_range - vol_regime) * 10))  # Smooth regime transition
    
    # Price efficiency measure - ratio of actual move to potential range
    price_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    efficiency_smooth = price_efficiency.rolling(window=5).mean()
    
    # Composite factor: Momentum synergy amplified by volume-amount confirmation,
    # filtered by volatility regime, and weighted by price efficiency
    factor = (
        momentum_synergy * 
        (1 + volume_amount_divergence) *  # Amplify or dampen based on divergence
        volatility_filter * 
        efficiency_smooth
    )
    
    return factor
