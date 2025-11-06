import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Regime-Adaptive Momentum with Volume Acceleration
    # Combines volatility-normalized momentum with asymmetric volume-weighting
    # Uses regime-dependent timeframes for adaptive signal alignment
    # Interpretable as: Stocks with strong momentum confirmed by accelerating volume in appropriate market regimes
    
    # Calculate returns and volatility
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=10).std()
    
    # Regime detection using volatility levels
    vol_regime = np.where(volatility > volatility.rolling(window=20).median(), 'high', 'low')
    
    # Regime-dependent momentum timeframes
    # Shorter timeframe for high volatility, longer for low volatility
    momentum_period = np.where(vol_regime == 'high', 5, 10)
    
    # Price momentum with regime-adaptive periods
    price_momentum = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= max(momentum_period):
            period = momentum_period[i]
            price_momentum.iloc[i] = (df['close'].iloc[i] - df['close'].iloc[i-period]) / df['close'].iloc[i-period]
    
    # Volume acceleration with asymmetric weighting
    # Positive acceleration gets higher weight than negative
    volume_acceleration = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)
    volume_weight = np.where(volume_acceleration > 0, 1.5, 0.8)  # Asymmetric weighting
    
    # Volatility-normalized momentum
    vol_normalized_mom = price_momentum / (volatility + 1e-7)
    
    # Directional convergence: price and volume alignment
    price_volume_alignment = np.sign(price_momentum) * np.sign(volume_acceleration)
    
    # Range-based confirmation
    daily_range = df['high'] - df['low']
    range_momentum = (daily_range - daily_range.shift(5)) / daily_range.shift(5)
    
    # Combine signals with economic rationale
    # Primary: volatility-normalized momentum
    # Secondary: volume acceleration with asymmetric weighting
    # Tertiary: directional convergence confirmation
    alpha = (
        0.5 * vol_normalized_mom +
        0.3 * (volume_acceleration * volume_weight) +
        0.2 * price_volume_alignment
    )
    
    # Simple smoothing for noise reduction (3-day moving average)
    alpha = alpha.rolling(window=3).mean()
    
    return alpha
