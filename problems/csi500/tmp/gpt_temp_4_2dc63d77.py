import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Weighted Regime Momentum factor
    Combines price momentum, regime detection, and volume divergence
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Calculate daily returns for volatility
    daily_returns = data['close'].pct_change()
    
    # Regime Detection Component
    # 50-day price trend
    trend_50d = (data['close'] / data['close'].shift(50)) - 1
    
    # 20-day volatility (standard deviation of daily returns)
    volatility_20d = daily_returns.rolling(window=20).std()
    
    # Classify regimes
    bull_regime = (trend_50d > 0.05) & (volatility_20d < 0.02)
    bear_regime = (trend_50d < -0.05) & (volatility_20d < 0.02)
    neutral_regime = ~bull_regime & ~bear_regime
    
    # Momentum Component
    # 10-day and 5-day momentum
    momentum_10d = (data['close'] / data['close'].shift(10)) - 1
    momentum_5d = (data['close'] / data['close'].shift(5)) - 1
    
    # Regime-dependent momentum weighting
    regime_momentum = pd.Series(index=data.index, dtype=float)
    regime_momentum[bull_regime] = 0.7 * momentum_10d[bull_regime] + 0.3 * momentum_5d[bull_regime]
    regime_momentum[bear_regime] = 0.3 * momentum_10d[bear_regime] + 0.7 * momentum_5d[bear_regime]
    regime_momentum[neutral_regime] = 0.5 * momentum_10d[neutral_regime] + 0.5 * momentum_5d[neutral_regime]
    
    # Volume Divergence Component
    # Calculate 5-day rolling percentile ranks
    volume_rank = data['volume'].rolling(window=5).apply(
        lambda x: (x.rank(pct=True).iloc[-1]), raw=False
    )
    price_change_rank = data['close'].pct_change(5).rolling(window=5).apply(
        lambda x: (x.rank(pct=True).iloc[-1]), raw=False
    )
    
    # Volume divergence
    volume_divergence = volume_rank - price_change_rank
    
    # Factor Integration
    # Scale momentum by regime strength
    scaled_momentum = regime_momentum * np.abs(trend_50d)
    
    # Multiply by volume divergence
    factor = scaled_momentum * volume_divergence
    
    return factor
