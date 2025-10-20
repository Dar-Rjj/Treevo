import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Divergence Momentum
    df['price_momentum_5'] = df['close'].pct_change(5)
    df['price_momentum_20'] = df['close'].pct_change(20)
    df['volume_momentum_5'] = df['volume'].pct_change(5)
    df['volume_momentum_20'] = df['volume'].pct_change(20)
    
    # Volume divergence: positive when volume and price move in opposite directions
    df['volume_divergence'] = np.where(
        df['price_momentum_5'] * df['volume_momentum_5'] < 0,
        abs(df['volume_momentum_5']),
        -abs(df['volume_momentum_5'])
    )
    
    # Volatility-Adjusted Return Reversal
    df['return_3d'] = df['close'].pct_change(3)
    df['volatility_20d'] = df['close'].pct_change().rolling(20).std()
    df['vol_adj_return'] = df['return_3d'] / (df['volatility_20d'] + 1e-8)
    
    # Intraday Strength Persistence
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['range_strength'] = df['daily_range'] / df['daily_range'].rolling(10).mean()
    
    # Count consecutive strong/weak days
    strong_days = (df['range_strength'] > 1.2).astype(int)
    weak_days = (df['range_strength'] < 0.8).astype(int)
    
    df['strong_streak'] = strong_days * (strong_days.groupby((strong_days != strong_days.shift()).cumsum()).cumcount() + 1)
    df['weak_streak'] = weak_days * (weak_days.groupby((weak_days != weak_days.shift()).cumsum()).cumcount() + 1)
    
    df['continuity_score'] = np.where(
        df['strong_streak'] > 0, 
        df['strong_streak'], 
        -df['weak_streak']
    )
    
    # Volume-Weighted Price Acceleration
    df['price_velocity'] = df['close'].pct_change()
    df['price_acceleration'] = df['price_velocity'].diff()
    df['volume_trend'] = df['volume'] / df['volume'].rolling(10).mean()
    
    df['vol_weighted_accel'] = df['price_acceleration'] * df['volume_trend']
    
    # Amplitude-Duration Momentum
    df['amplitude'] = (df['high'] - df['low']) / df['close']
    df['duration_score'] = df['amplitude'].rolling(5).std()
    
    # Recent weighting using exponential decay
    weights = np.exp(-np.arange(5) / 2.0)
    weights = weights / weights.sum()
    
    df['weighted_amplitude'] = df['amplitude'].rolling(5).apply(
        lambda x: np.sum(x * weights) if len(x) == 5 else np.nan
    )
    
    df['composite_score'] = df['weighted_amplitude'] * df['duration_score']
    
    # Combine all factors with weights
    factor = (
        0.3 * df['volume_divergence'] +
        0.25 * df['vol_adj_return'] +
        0.2 * df['continuity_score'] +
        0.15 * df['vol_weighted_accel'] +
        0.1 * df['composite_score']
    )
    
    return factor
