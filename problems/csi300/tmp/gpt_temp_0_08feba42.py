import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Regime-Adaptive Momentum with Volume Acceleration
    # Combines volatility-normalized momentum signals with volume acceleration and regime detection
    # Interpretable as: Stocks with strong momentum across price and volume dimensions, weighted by market regime
    
    # Volatility calculation using 10-day rolling standard deviation of returns
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=10).std()
    
    # Regime detection using rolling volatility percentiles
    vol_regime = volatility.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # Adaptive window selection based on volatility regime
    # High volatility: shorter windows, Low volatility: longer windows
    momentum_window = np.where(vol_regime > 0.7, 5, 
                              np.where(vol_regime < 0.3, 10, 7))
    
    # Price momentum with adaptive windows
    price_momentum = pd.Series(index=df.index, dtype=float)
    for i, window in enumerate(momentum_window):
        if i >= window:
            price_momentum.iloc[i] = (df['close'].iloc[i] - df['close'].iloc[i-int(window)]) / df['close'].iloc[i-int(window)]
    
    # Volume acceleration (rate of change in volume momentum)
    volume_momentum = df['volume'].pct_change(periods=3)
    volume_acceleration = volume_momentum - volume_momentum.shift(3)
    
    # Range-based momentum efficiency
    true_range = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    range_efficiency = (df['close'] - df['close'].shift(1)) / (true_range + 1e-7)
    
    # Volume-weighted price momentum
    volume_weight = df['volume'] / df['volume'].rolling(window=20).mean()
    volume_weighted_momentum = price_momentum * volume_weight
    
    # Volatility-normalized components
    vol_normalized_momentum = price_momentum / (volatility + 1e-7)
    vol_normalized_range = range_efficiency.rolling(window=5).mean() / (volatility + 1e-7)
    
    # Multiplicative volume-weighting for acceleration
    volume_acceleration_weighted = volume_acceleration * np.abs(price_momentum)
    
    # Regime-adaptive signal blending
    # High volatility: emphasize volatility-normalized signals
    # Low volatility: emphasize raw momentum and volume acceleration
    high_vol_weight = np.where(vol_regime > 0.7, 0.6, 0.3)
    low_vol_weight = np.where(vol_regime < 0.3, 0.5, 0.2)
    
    alpha = (
        high_vol_weight * vol_normalized_momentum +
        (1 - high_vol_weight - low_vol_weight) * volume_weighted_momentum +
        low_vol_weight * volume_acceleration_weighted +
        0.2 * vol_normalized_range
    )
    
    return alpha
