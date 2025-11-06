import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Momentum Divergence factor that captures relative momentum strength
    and divergence patterns between stock and sector/industry performance.
    """
    # Calculate stock momentum metrics
    df = df.copy()
    
    # Stock momentum calculations
    df['stock_ret_10d'] = df['close'] / df['close'].shift(10) - 1
    df['stock_ret_5d'] = df['close'] / df['close'].shift(5) - 1
    df['stock_mom_accel_10d'] = df['stock_ret_10d'] - df['stock_ret_10d'].shift(5)
    df['stock_mom_accel_5d'] = df['stock_ret_5d'] - df['stock_ret_5d'].shift(3)
    
    # Calculate sector momentum (using rolling median as proxy for sector)
    df['sector_ret_10d'] = df['close'].rolling(window=10).apply(lambda x: np.median(x.pct_change().dropna()), raw=False)
    df['sector_ret_5d'] = df['close'].rolling(window=5).apply(lambda x: np.median(x.pct_change().dropna()), raw=False)
    df['sector_mom_accel_10d'] = df['sector_ret_10d'] - df['sector_ret_10d'].shift(5)
    df['sector_mom_accel_5d'] = df['sector_ret_5d'] - df['sector_ret_5d'].shift(3)
    
    # Relative momentum strength
    df['rel_mom_strength_10d'] = df['stock_ret_10d'] - df['sector_ret_10d']
    df['rel_mom_strength_5d'] = df['stock_ret_5d'] - df['sector_ret_5d']
    
    # Momentum divergence detection
    df['divergence_10d'] = np.where(
        (df['stock_mom_accel_10d'] > 0) & (df['sector_mom_accel_10d'] < 0), 1,
        np.where((df['stock_mom_accel_10d'] < 0) & (df['sector_mom_accel_10d'] > 0), -1, 0)
    )
    
    df['divergence_5d'] = np.where(
        (df['stock_mom_accel_5d'] > 0) & (df['sector_mom_accel_5d'] < 0), 1,
        np.where((df['stock_mom_accel_5d'] < 0) & (df['sector_mom_accel_5d'] > 0), -1, 0)
    )
    
    # Divergence magnitude scoring
    df['divergence_magnitude_10d'] = np.abs(df['stock_mom_accel_10d'] - df['sector_mom_accel_10d'])
    df['divergence_magnitude_5d'] = np.abs(df['stock_mom_accel_5d'] - df['sector_mom_accel_5d'])
    
    # Directional consistency over 3-day window
    df['divergence_consistency'] = (
        df['divergence_10d'].rolling(window=3).sum() + 
        df['divergence_5d'].rolling(window=3).sum()
    ) / 6
    
    # Composite momentum divergence factor
    df['momentum_divergence_factor'] = (
        df['rel_mom_strength_10d'].rank(pct=True) * 0.3 +
        df['rel_mom_strength_5d'].rank(pct=True) * 0.3 +
        df['divergence_consistency'].rank(pct=True) * 0.2 +
        (df['divergence_magnitude_10d'] * df['divergence_10d']).rank(pct=True) * 0.1 +
        (df['divergence_magnitude_5d'] * df['divergence_5d']).rank(pct=True) * 0.1
    )
    
    return df['momentum_divergence_factor']
