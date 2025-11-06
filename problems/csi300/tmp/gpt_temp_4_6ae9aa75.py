import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-Normalized Momentum with Volume Acceleration Factor
    # Combines regime-adaptive momentum signals with volume acceleration and multiplicative weighting
    # Interpretable as: Stocks with strong momentum relative to volatility, confirmed by accelerating volume
    
    # Calculate returns and volatility
    returns = df['close'].pct_change()
    
    # Adaptive window selection based on recent volatility regime
    vol_5d = returns.rolling(window=5).std()
    vol_20d = returns.rolling(window=20).std()
    vol_regime = vol_5d / (vol_20d + 1e-7)
    
    # Short window for high volatility, longer for low volatility
    momentum_window = np.where(vol_regime > 1.2, 3, 
                              np.where(vol_regime < 0.8, 10, 5))
    
    # Volatility-normalized momentum with adaptive windows
    price_momentum = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= max(momentum_window):
            window = int(momentum_window[i])
            ret = (df['close'].iloc[i] - df['close'].iloc[i-window]) / df['close'].iloc[i-window]
            vol = returns.iloc[i-window:i+1].std()
            price_momentum.iloc[i] = ret / (vol + 1e-7)
    
    # Volume acceleration with regime context
    volume_accel = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 10:
            short_vol = df['volume'].iloc[i-2:i+1].mean()
            long_vol = df['volume'].iloc[i-10:i+1].mean()
            volume_accel.iloc[i] = (short_vol - long_vol) / (long_vol + 1e-7)
    
    # Range-based momentum confirmation
    range_momentum = (df['high'] - df['low'].shift(1)) / (df['low'].shift(1) + 1e-7)
    range_momentum_vol_adj = range_momentum / (vol_5d + 1e-7)
    
    # Multiplicative volume-weighting factor
    volume_weight = (df['volume'] / df['volume'].rolling(window=20).mean())
    
    # Combine components with volume acceleration emphasis
    alpha = (
        price_momentum * 
        (1 + 0.5 * volume_accel.fillna(0)) *  # Volume acceleration multiplier
        volume_weight *  # Volume weighting
        (1 + 0.3 * np.tanh(range_momentum_vol_adj))  # Range momentum enhancement
    )
    
    return alpha
