import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Price-Volume Efficiency Momentum Divergence factor
    Combines price efficiency, volume efficiency, and their divergence patterns
    """
    df = data.copy()
    
    # Price efficiency calculations
    df['daily_range'] = df['high'] - df['low']
    df['daily_return'] = df['close'].pct_change()
    df['price_efficiency'] = df['daily_return'].abs() / (df['daily_range'] + 1e-8)
    
    # Volume efficiency calculations
    df['volume_efficiency'] = df['volume'] / (df['daily_range'] + 1e-8)
    
    # 5-day and 21-day moving averages for efficiency metrics
    df['price_eff_5d'] = df['price_efficiency'].rolling(window=5, min_periods=3).mean()
    df['price_eff_21d'] = df['price_efficiency'].rolling(window=21, min_periods=10).mean()
    df['volume_eff_5d'] = df['volume_efficiency'].rolling(window=5, min_periods=3).mean()
    df['volume_eff_21d'] = df['volume_efficiency'].rolling(window=21, min_periods=10).mean()
    
    # Efficiency momentum calculations
    df['price_eff_momentum_5d'] = df['price_eff_5d'] / df['price_eff_5d'].shift(5) - 1
    df['price_eff_momentum_21d'] = df['price_eff_5d'] / df['price_eff_21d'] - 1
    
    df['volume_eff_momentum_5d'] = df['volume_eff_5d'] / df['volume_eff_5d'].shift(5) - 1
    df['volume_eff_momentum_21d'] = df['volume_eff_5d'] / df['volume_eff_21d'] - 1
    
    # Range momentum
    df['range_momentum_5d'] = df['daily_range'].rolling(window=5, min_periods=3).mean() / \
                             df['daily_range'].rolling(window=10, min_periods=5).mean() - 1
    
    # Efficiency divergence detection
    df['eff_divergence'] = 0
    price_up = df['price_eff_momentum_5d'] > 0
    volume_down = df['volume_eff_momentum_5d'] < 0
    price_down = df['price_eff_momentum_5d'] < 0
    volume_up = df['volume_eff_momentum_5d'] > 0
    
    df.loc[price_up & volume_down, 'eff_divergence'] = 1  # Positive divergence
    df.loc[price_down & volume_up, 'eff_divergence'] = -1  # Negative divergence
    
    # Divergence strength
    df['divergence_strength'] = (df['price_eff_momentum_5d'].abs() * df['volume_eff_momentum_5d'].abs()) * df['eff_divergence']
    
    # Multi-timeframe divergence
    df['multi_tf_divergence'] = np.where(
        (df['price_eff_momentum_5d'] * df['price_eff_momentum_21d'] < 0) & 
        (df['volume_eff_momentum_5d'] * df['volume_eff_momentum_21d'] > 0),
        df['price_eff_momentum_5d'] - df['price_eff_momentum_21d'],
        0
    )
    
    # Efficiency regime classification
    df['efficiency_regime'] = np.where(
        df['price_efficiency'] > df['price_efficiency'].rolling(window=20, min_periods=10).mean(),
        1,  # High efficiency
        -1   # Low efficiency
    )
    
    # Persistence analysis
    df['price_eff_persistence'] = 0
    df['volume_eff_persistence'] = 0
    
    # Count consecutive days of efficiency improvement
    for i in range(1, len(df)):
        if df['price_eff_momentum_5d'].iloc[i] > df['price_eff_momentum_5d'].iloc[i-1]:
            df['price_eff_persistence'].iloc[i] = df['price_eff_persistence'].iloc[i-1] + 1
        elif df['price_eff_momentum_5d'].iloc[i] < df['price_eff_momentum_5d'].iloc[i-1]:
            df['price_eff_persistence'].iloc[i] = df['price_eff_persistence'].iloc[i-1] - 1
        
        if df['volume_eff_momentum_5d'].iloc[i] > df['volume_eff_momentum_5d'].iloc[i-1]:
            df['volume_eff_persistence'].iloc[i] = df['volume_eff_persistence'].iloc[i-1] + 1
        elif df['volume_eff_momentum_5d'].iloc[i] < df['volume_eff_momentum_5d'].iloc[i-1]:
            df['volume_eff_persistence'].iloc[i] = df['volume_eff_persistence'].iloc[i-1] - 1
    
    # Cross-efficiency persistence alignment
    df['cross_persistence'] = df['price_eff_persistence'] * df['volume_eff_persistence']
    
    # Regime transition analysis
    df['regime_change'] = df['efficiency_regime'].diff()
    df['regime_momentum'] = df['regime_change'].rolling(window=5, min_periods=3).sum()
    
    # Composite factor generation
    df['composite_factor'] = (
        df['price_eff_momentum_5d'] * 0.3 +
        df['volume_eff_momentum_5d'] * 0.2 +
        df['divergence_strength'] * 0.25 +
        df['multi_tf_divergence'] * 0.15 +
        df['cross_persistence'] * 0.1
    )
    
    # Range-adjusted final factor
    df['final_factor'] = df['composite_factor'] / (df['daily_range'].rolling(window=10, min_periods=5).mean() + 1e-8)
    
    # Normalize by recent volatility
    factor_vol = df['final_factor'].rolling(window=20, min_periods=10).std()
    df['normalized_factor'] = df['final_factor'] / (factor_vol + 1e-8)
    
    return df['normalized_factor']
