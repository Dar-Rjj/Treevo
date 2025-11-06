import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Regime-Adaptive Volatility-Normalized Momentum with Volume Acceleration
    # Combines volatility-normalized momentum signals with volume acceleration and regime-adaptive windows
    # Interpretable as: Stocks showing strong momentum relative to volatility, confirmed by accelerating volume
    
    # Calculate returns for momentum and volatility
    returns = df['close'].pct_change()
    
    # Regime-adaptive window selection based on recent volatility
    recent_vol = returns.rolling(window=10).std()
    vol_regime = np.where(recent_vol > recent_vol.rolling(window=20).median(), 'high', 'low')
    
    # Adaptive momentum windows: shorter in high vol, longer in low vol
    momentum_window = np.where(vol_regime == 'high', 5, 10)
    
    # Volatility-normalized momentum with adaptive windows
    volatility = returns.rolling(window=20).std()
    
    # Price momentum normalized by volatility
    price_momentum = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= momentum_window[i]:
            window = int(momentum_window[i])
            ret = (df['close'].iloc[i] - df['close'].iloc[i - window]) / df['close'].iloc[i - window]
            price_momentum.iloc[i] = ret / (volatility.iloc[i] + 1e-7)
        else:
            price_momentum.iloc[i] = 0
    
    # Volume acceleration with multiplicative weighting
    volume_accel = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= momentum_window[i]:
            window = int(momentum_window[i])
            current_vol = df['volume'].iloc[i]
            past_vol = df['volume'].iloc[i - window]
            vol_growth = (current_vol - past_vol) / (past_vol + 1e-7)
            volume_accel.iloc[i] = vol_growth * np.sign(price_momentum.iloc[i])
        else:
            volume_accel.iloc[i] = 0
    
    # Range-based momentum confirmation
    range_momentum = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= momentum_window[i]:
            window = int(momentum_window[i])
            current_range = df['high'].iloc[i] - df['low'].iloc[i]
            past_range = df['high'].iloc[i - window] - df['low'].iloc[i - window]
            range_change = (current_range - past_range) / (past_range + 1e-7)
            range_momentum.iloc[i] = range_change / (volatility.iloc[i] + 1e-7)
        else:
            range_momentum.iloc[i] = 0
    
    # Multiplicative volume-weighting factor
    volume_weight = np.log1p(df['volume'] / df['volume'].rolling(window=20).mean())
    
    # Combine signals with volume weighting
    alpha = (
        price_momentum * 0.5 +
        volume_accel * 0.3 +
        range_momentum * 0.2
    ) * volume_weight
    
    return alpha
