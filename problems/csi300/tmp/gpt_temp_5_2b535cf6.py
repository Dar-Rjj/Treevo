import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Multi-Timeframe Momentum factor
    Combines volatility regime classification with multi-timeframe momentum signals
    """
    df = data.copy()
    
    # Calculate daily returns for volatility calculation
    df['returns'] = df['close'].pct_change()
    
    # Volatility-Regime Classification
    # 20-day rolling volatility (standard deviation of daily returns)
    df['volatility_20d'] = df['returns'].rolling(window=20, min_periods=15).std()
    
    # Calculate rolling percentiles for volatility regime classification
    df['volatility_percentile'] = df['volatility_20d'].rolling(window=252, min_periods=200).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 60)) * 2 + 
                 (x.iloc[-1] < np.percentile(x.dropna(), 40)) * 1, 
        raw=False
    )
    
    # Regime assignment: 2=High, 1=Low, 0=Normal
    df['volatility_regime'] = np.where(df['volatility_percentile'] == 2, 'high',
                                      np.where(df['volatility_percentile'] == 1, 'low', 'normal'))
    
    # Multi-Timeframe Momentum Signals
    # Short-term momentum (3-day)
    df['price_momentum_3d'] = df['close'] / df['close'].shift(3) - 1
    df['volume_momentum_3d'] = df['volume'] / df['volume'].shift(3) - 1
    
    # Medium-term momentum (10-day)
    df['price_momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['volume_momentum_10d'] = df['volume'] / df['volume'].shift(10) - 1
    
    # Long-term momentum (20-day)
    df['price_momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    df['volume_momentum_20d'] = df['volume'] / df['volume'].shift(20) - 1
    
    # Combined momentum signals (price + volume)
    df['momentum_short'] = df['price_momentum_3d'] + df['volume_momentum_3d']
    df['momentum_medium'] = df['price_momentum_10d'] + df['volume_momentum_10d']
    df['momentum_long'] = df['price_momentum_20d'] + df['volume_momentum_20d']
    
    # Regime-Adaptive Signal Blending
    def regime_blending(row):
        if row['volatility_regime'] == 'high':
            # High volatility: emphasize short-term signals (60% short, 30% medium, 10% long)
            return (0.6 * row['momentum_short'] + 
                    0.3 * row['momentum_medium'] + 
                    0.1 * row['momentum_long'])
        elif row['volatility_regime'] == 'low':
            # Low volatility: emphasize medium-term signals (30% short, 50% medium, 20% long)
            return (0.3 * row['momentum_short'] + 
                    0.5 * row['momentum_medium'] + 
                    0.2 * row['momentum_long'])
        else:
            # Normal volatility: balanced approach (40% short, 40% medium, 20% long)
            return (0.4 * row['momentum_short'] + 
                    0.4 * row['momentum_medium'] + 
                    0.2 * row['momentum_long'])
    
    # Apply regime-adaptive blending
    df['regime_adaptive_momentum'] = df.apply(regime_blending, axis=1)
    
    # Final factor: regime-adaptive momentum normalized by recent volatility
    factor = df['regime_adaptive_momentum'] / (df['volatility_20d'] + 1e-8)
    
    return factor
