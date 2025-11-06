import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining multi-timeframe price-volume convergence,
    regime-adaptive range efficiency, volume-confirmed extreme reversal, and adaptive flow momentum.
    """
    # Price Momentum Hierarchy
    df['momentum_3d'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_acceleration'] = (df['momentum_3d'] / df['momentum_5d']) / (df['momentum_5d'] / df['momentum_10d'])
    
    # Volume Confirmation Framework
    df['volume_momentum'] = (df['volume'] / df['volume'].shift(3)) / (df['volume'].shift(3) / df['volume'].shift(6))
    df['volume_persistence'] = df['volume'].rolling(window=5).apply(lambda x: (x > x.shift(1)).sum(), raw=False)
    
    # Multi-timeframe momentum alignment
    df['momentum_alignment'] = (
        (df['momentum_3d'] > 0).astype(int) + 
        (df['momentum_5d'] > 0).astype(int) + 
        (df['momentum_10d'] > 0).astype(int)
    )
    
    # Volume confirmation strength (rolling correlation)
    df['volume_confirmation'] = df['momentum_3d'].rolling(window=10).corr(df['volume_momentum'])
    
    # Regime-Adaptive Range Efficiency
    df['range_volatility'] = (df['high'].rolling(window=10).max() - df['low'].rolling(window=10).min()) / df['close'].shift(10)
    df['volatility_momentum'] = df['range_volatility'].rolling(window=5).mean() / df['range_volatility'].rolling(window=10).mean()
    
    # Volatility regime classification
    vol_regime_percentiles = df['range_volatility'].rolling(window=50).apply(
        lambda x: pd.cut([x.iloc[-1]], bins=[0, 0.33, 0.67, 1], labels=[0, 1, 2])[0], raw=False
    )
    
    # Efficiency metrics
    df['single_day_efficiency'] = abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    df['multi_day_efficiency'] = abs(df['close'] - df['close'].shift(3)) / (
        df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min()
    )
    
    # Volume-Confirmed Extreme Reversal
    df['price_extreme'] = (df['close'] - df['close'].rolling(window=5).mean()) / (
        df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()
    )
    df['volume_extreme'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['range_extreme'] = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=5).mean()
    
    # Extreme confirmation
    df['extreme_confirmation'] = df['volume_extreme'] * df['price_extreme']
    df['extreme_dimensions'] = (
        (abs(df['price_extreme']) > 1).astype(int) + 
        (df['volume_extreme'] > 2).astype(int) + 
        (df['range_extreme'] > 1.5).astype(int)
    )
    
    # Adaptive Flow Momentum
    df['directional_flow'] = np.sign(df['close'] - df['close'].shift(1)) * df['amount']
    df['flow_momentum'] = (df['directional_flow'] / df['directional_flow'].shift(3)) / (
        df['directional_flow'].shift(3) / df['directional_flow'].shift(6)
    )
    df['flow_persistence'] = df['directional_flow'].rolling(window=5).apply(
        lambda x: (np.sign(x) == np.sign(x.shift(1))).sum(), raw=False
    )
    
    # Flow efficiency and integration
    df['flow_efficiency'] = abs(df['directional_flow']) / df['volume']
    df['flow_volume_divergence'] = df['flow_momentum'] - df['volume_momentum']
    
    # Composite factor calculation
    factor = (
        # Multi-timeframe convergence (40% weight)
        0.4 * (
            df['momentum_alignment'] * df['volume_confirmation'].fillna(0) * 
            np.sign(df['momentum_acceleration']).fillna(0)
        ) +
        # Regime-adaptive efficiency (25% weight)
        0.25 * (
            df['multi_day_efficiency'] * (1 - vol_regime_percentiles / 2) * 
            df['single_day_efficiency']
        ) +
        # Extreme reversal (20% weight)
        0.2 * (
            -df['extreme_confirmation'] * df['extreme_dimensions'] * 
            np.sign(df['price_extreme'])
        ) +
        # Flow momentum (15% weight)
        0.15 * (
            df['flow_persistence'] * df['flow_efficiency'] * 
            np.sign(df['flow_momentum']).fillna(0)
        )
    )
    
    return factor
