import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Multi-Timeframe Momentum Structure
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_10d'] = df['close'].pct_change(10)
    df['momentum_20d'] = df['close'].pct_change(20)
    df['momentum_accel'] = df['momentum_5d'] - df['momentum_10d']
    
    # Volume Confirmation Patterns
    df['volume_momentum'] = df['volume'].pct_change(5)
    df['volume_adj_range'] = (df['high'] - df['low']) * df['volume']
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
    df['volume_direction'] = np.sign(df['returns']) * np.sign(df['volume'].pct_change())
    
    # Volatility Context
    df['volatility_20d'] = df['returns'].rolling(20).std()
    df['true_range'] = np.maximum(df['high'] - df['low'], 
                                 np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                           abs(df['low'] - df['close'].shift(1))))
    df['volatility_regime'] = (df['volatility_20d'] > df['volatility_20d'].rolling(60).median()).astype(int)
    
    # Multi-dimensional Divergence Detection
    df['price_volume_divergence'] = np.sign(df['momentum_5d']) * np.sign(df['volume_momentum'])
    df['accel_volume_alignment'] = np.sign(df['momentum_accel']) * np.sign(df['volume_momentum'])
    
    # Momentum divergence across timeframes
    df['momentum_divergence'] = (
        (np.sign(df['momentum_5d']) != np.sign(df['momentum_10d'])).astype(int) +
        (np.sign(df['momentum_10d']) != np.sign(df['momentum_20d'])).astype(int) +
        (np.sign(df['momentum_5d']) != np.sign(df['momentum_20d'])).astype(int)
    )
    
    df['range_confirmation'] = df['volume_adj_range'] / df['volume_adj_range'].rolling(10).mean()
    
    # Regime-Specific Adjustments
    # High volatility regime: emphasize momentum continuation
    high_vol_factor = df['momentum_5d'] * (1 + df['volatility_regime'] * 0.5)
    
    # Low volatility regime: amplify mean reversion
    low_vol_factor = -df['momentum_divergence'] * df['momentum_5d'] * (1 - df['volatility_regime'])
    
    # Volume confirmation scaling
    volume_factor = df['momentum_5d'] * df['volume_ratio'] * df['price_volume_divergence']
    
    # Range adjustment
    range_factor = df['momentum_accel'] * df['range_confirmation']
    
    # Generate Composite Alpha Factor
    # Strong trend continuation
    trend_continuation = (
        (df['price_volume_divergence'] > 0) & 
        (df['accel_volume_alignment'] > 0) &
        (df['volatility_regime'] == 1)
    ) * (high_vol_factor + volume_factor)
    
    # Mean reversion setup
    mean_reversion = (
        (df['momentum_divergence'] >= 2) & 
        (df['price_volume_divergence'] < 0) &
        (df['volatility_regime'] == 0)
    ) * low_vol_factor
    
    # Breakout potential
    breakout = (
        (df['momentum_accel'] > 0) & 
        (df['volume_ratio'] > 1.5) &
        (df['range_confirmation'] > 1.2)
    ) * (range_factor + volume_factor)
    
    # Weak signal filtering
    weak_filter = (
        (df['momentum_divergence'] == 1) & 
        (abs(df['price_volume_divergence']) < 0.1) &
        (df['volume_ratio'].between(0.8, 1.2))
    )
    
    # Composite factor with regime adjustments
    composite_factor = (
        trend_continuation.fillna(0) + 
        mean_reversion.fillna(0) + 
        breakout.fillna(0)
    )
    
    # Apply weak signal filter (reduce magnitude)
    composite_factor = np.where(weak_filter, composite_factor * 0.3, composite_factor)
    
    # Final normalization
    composite_factor = composite_factor / composite_factor.rolling(20).std()
    
    return pd.Series(composite_factor, index=df.index)
