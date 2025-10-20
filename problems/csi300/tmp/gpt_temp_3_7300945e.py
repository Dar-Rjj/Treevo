import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Efficiency Momentum Divergence factor that combines:
    - Price efficiency momentum across multiple timeframes
    - Volume efficiency patterns and acceleration
    - Price-volume efficiency divergence detection
    - Range-based momentum integration
    - Efficiency persistence analysis
    """
    
    # Price efficiency calculations
    df['daily_range'] = df['high'] - df['low']
    df['price_efficiency_5d'] = (df['close'] - df['close'].shift(5)) / (df['daily_range'].rolling(5).mean().replace(0, np.nan))
    df['price_efficiency_10d'] = (df['close'] - df['close'].shift(10)) / (df['daily_range'].rolling(10).mean().replace(0, np.nan))
    df['price_efficiency_21d'] = (df['close'] - df['close'].shift(21)) / (df['daily_range'].rolling(21).mean().replace(0, np.nan))
    
    # Price efficiency momentum and acceleration
    df['price_eff_momentum_5d'] = df['price_efficiency_5d'] - df['price_efficiency_5d'].shift(5)
    df['price_eff_momentum_10d'] = df['price_efficiency_10d'] - df['price_efficiency_10d'].shift(10)
    df['price_eff_accel_5v21'] = df['price_efficiency_5d'] - df['price_efficiency_21d']
    df['price_eff_accel_5v10'] = df['price_efficiency_5d'] - df['price_efficiency_10d']
    
    # Volume efficiency calculations
    df['volume_efficiency'] = df['volume'] / df['daily_range'].replace(0, np.nan)
    df['vol_eff_5d'] = df['volume_efficiency'].rolling(5).mean()
    df['vol_eff_10d'] = df['volume_efficiency'].rolling(10).mean()
    df['vol_eff_21d'] = df['volume_efficiency'].rolling(21).mean()
    
    # Volume efficiency momentum and acceleration
    df['vol_eff_momentum_5d'] = df['vol_eff_5d'] - df['vol_eff_5d'].shift(5)
    df['vol_eff_momentum_10d'] = df['vol_eff_10d'] - df['vol_eff_10d'].shift(10)
    df['vol_eff_accel_5v21'] = df['vol_eff_5d'] - df['vol_eff_21d']
    df['vol_eff_accel_5v10'] = df['vol_eff_5d'] - df['vol_eff_10d']
    
    # Price-volume efficiency divergence
    df['price_vol_divergence'] = (
        (df['price_eff_momentum_5d'] > 0) & (df['vol_eff_momentum_5d'] < 0) - 
        (df['price_eff_momentum_5d'] < 0) & (df['vol_eff_momentum_5d'] > 0)
    )
    
    # Multi-timeframe divergence patterns
    df['multi_tf_divergence'] = (
        (df['price_eff_accel_5v21'] * df['vol_eff_accel_5v21'] < 0).astype(int) * 
        np.sign(df['price_eff_accel_5v21'])
    )
    
    # Range-based momentum
    df['range_momentum_5d'] = df['daily_range'].pct_change(5)
    df['range_momentum_10d'] = df['daily_range'].pct_change(10)
    df['range_momentum_21d'] = df['daily_range'].pct_change(21)
    
    # Range-volume interaction
    df['range_vol_interaction'] = (
        df['range_momentum_5d'] * df['vol_eff_momentum_5d'] * 
        np.where(df['range_momentum_5d'] > 0, 1, -1)
    )
    
    # Efficiency persistence analysis
    df['price_eff_persistence'] = (
        (df['price_eff_momentum_5d'] > 0).rolling(5).sum() - 
        (df['price_eff_momentum_5d'] < 0).rolling(5).sum()
    )
    
    df['vol_eff_persistence'] = (
        (df['vol_eff_momentum_5d'] > 0).rolling(5).sum() - 
        (df['vol_eff_momentum_5d'] < 0).rolling(5).sum()
    )
    
    # Cross-efficiency persistence alignment
    df['cross_eff_persistence'] = (
        df['price_eff_persistence'] * df['vol_eff_persistence'] * 
        np.where(df['price_eff_persistence'] == df['vol_eff_persistence'], 1, -1)
    )
    
    # Efficiency regime classification
    df['high_efficiency_regime'] = (df['price_efficiency_5d'] > df['price_efficiency_21d']).astype(int)
    df['low_efficiency_regime'] = (df['price_efficiency_5d'] < df['price_efficiency_21d']).astype(int)
    
    # Strong efficiency momentum signal
    df['strong_eff_momentum'] = (
        (df['price_eff_momentum_5d'] > 0) & 
        (df['price_eff_momentum_10d'] > 0) & 
        (df['vol_eff_momentum_5d'] > 0) & 
        (df['range_momentum_5d'] > 0)
    ).astype(int)
    
    # Efficiency divergence reversal signal
    df['eff_divergence_reversal'] = (
        (df['price_vol_divergence'] != 0) & 
        (df['multi_tf_divergence'] != 0) & 
        (df['range_momentum_5d'] < 0) & 
        (df['cross_eff_persistence'] < 0)
    ).astype(int) * np.sign(df['price_vol_divergence'])
    
    # Range-adjusted efficiency weighting
    df['range_adjusted_weight'] = df['daily_range'] / df['daily_range'].rolling(21).mean()
    
    # Composite efficiency divergence factor
    composite_factor = (
        # Price efficiency components
        0.25 * df['price_eff_momentum_5d'].rank(pct=True) +
        0.15 * df['price_eff_accel_5v21'].rank(pct=True) +
        
        # Volume efficiency components
        0.20 * df['vol_eff_momentum_5d'].rank(pct=True) +
        0.10 * df['vol_eff_accel_5v21'].rank(pct=True) +
        
        # Divergence patterns
        0.15 * df['price_vol_divergence'].rank(pct=True) +
        0.10 * df['multi_tf_divergence'].rank(pct=True) +
        
        # Persistence and regime
        0.05 * df['cross_eff_persistence'].rank(pct=True) +
        
        # Range-adjusted signals
        df['range_adjusted_weight'] * (
            0.05 * df['strong_eff_momentum'].rank(pct=True) +
            0.05 * df['eff_divergence_reversal'].rank(pct=True)
        )
    )
    
    # Final factor with normalization
    factor = composite_factor.rolling(5).mean()
    factor = (factor - factor.rolling(63).mean()) / factor.rolling(63).std()
    
    return factor
