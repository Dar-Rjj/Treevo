import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Aware Momentum Acceleration factor
    Combines momentum acceleration with amount-based regime detection
    """
    df = data.copy()
    
    # Price Momentum Series
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # Momentum Acceleration
    df['accel_5d'] = (df['momentum_5d'] / df['momentum_5d'].shift(1)) - 1
    df['accel_10d'] = (df['momentum_10d'] / df['momentum_10d'].shift(1)) - 1
    df['accel_20d'] = (df['momentum_20d'] / df['momentum_20d'].shift(1)) - 1
    
    # Regime Detection
    df['amount_accel_20d'] = (df['amount'] / df['amount'].shift(20)) - 1
    df['high_participation'] = df['amount_accel_20d'] > 0
    
    # Market Condition Assessment
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['avg_true_range_20d'] = df['true_range'].rolling(window=20).mean()
    df['trending_market'] = df['avg_true_range_20d'] > df['avg_true_range_20d'].rolling(window=60).mean()
    
    # Exponential Smoothing
    alpha = 0.3
    df['smooth_accel_5d'] = df['accel_5d'].ewm(alpha=alpha).mean()
    df['smooth_accel_10d'] = df['accel_10d'].ewm(alpha=alpha).mean()
    df['smooth_accel_20d'] = df['accel_20d'].ewm(alpha=alpha).mean()
    
    # Regime-Weighted Combination
    def regime_weighted_combination(row):
        if row['high_participation']:
            # High participation: emphasize shorter-term acceleration
            weights = [0.5, 0.35, 0.15]  # 5d, 10d, 20d
        else:
            # Low participation: emphasize longer-term acceleration
            weights = [0.2, 0.3, 0.5]  # 5d, 10d, 20d
        
        weighted_accel = (weights[0] * row['smooth_accel_5d'] + 
                         weights[1] * row['smooth_accel_10d'] + 
                         weights[2] * row['smooth_accel_20d'])
        
        # Trending market enhancement
        if row['trending_market']:
            weighted_accel *= 1.2
        
        # Amount confirmation
        amount_sign = 1 if row['amount_accel_20d'] > 0 else 0.8
        weighted_accel *= amount_sign
        
        return weighted_accel
    
    df['regime_weighted_accel'] = df.apply(regime_weighted_combination, axis=1)
    
    # Cross-sectional ranking within each date
    def cross_sectional_ranking(group):
        ranked = group.rank(pct=True) - 0.5  # Center around 0
        # Identify outliers (top/bottom 10%)
        outlier_mask = (ranked > 0.4) | (ranked < -0.4)
        # Enhance outlier signals
        ranked[outlier_mask] *= 1.5
        return ranked
    
    # Apply cross-sectional ranking
    factor_series = df.groupby(df.index)['regime_weighted_accel'].transform(cross_sectional_ranking)
    
    return factor_series
