import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    import pandas as pd
    import numpy as np
    
    # Multi-timeframe momentum
    momentum_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Volatility regime detection (20-day rolling std of returns)
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=20, min_periods=10).std()
    vol_threshold = volatility.median()
    
    # Regime-based momentum weighting
    high_vol_regime = volatility > vol_threshold
    momentum_blend = pd.Series(index=df.index, dtype=float)
    momentum_blend[high_vol_regime] = 0.7 * momentum_1d + 0.3 * momentum_5d
    momentum_blend[~high_vol_regime] = 0.3 * momentum_1d + 0.7 * momentum_5d
    
    # Volume trend normalization (10-day linear trend)
    volume_trend = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 9:
            window_volume = df['volume'].iloc[i-9:i+1].values
            if len(window_volume) == 10:
                x = np.arange(10)
                slope = np.polyfit(x, window_volume, 1)[0]
                volume_trend.iloc[i] = slope / (window_volume.mean() + 1e-7)
    
    # Final factor: regime-weighted momentum scaled by volume trend
    factor = momentum_blend * volume_trend
    
    return factor
