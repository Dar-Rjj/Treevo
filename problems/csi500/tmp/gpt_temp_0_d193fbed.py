import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate basic price and volume features
    df = df.copy()
    
    # True range calculation
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Price momentum calculations
    df['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['price_momentum_3'] = df['close'] / df['close'].shift(3) - 1
    df['price_momentum_8'] = df['close'] / df['close'].shift(8) - 1
    df['price_acceleration'] = df['price_momentum_3'] - df['price_momentum_8']
    
    # Volume momentum calculations
    df['volume_momentum_5'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_momentum_3'] = df['volume'] / df['volume'].shift(3) - 1
    df['volume_momentum_8'] = df['volume'] / df['volume'].shift(8) - 1
    df['volume_acceleration'] = df['volume_momentum_3'] - df['volume_momentum_8']
    
    # Range momentum calculations
    df['range_momentum_5'] = df['true_range'] / df['true_range'].shift(5) - 1
    df['range_momentum_3'] = df['true_range'] / df['true_range'].shift(3) - 1
    df['range_momentum_8'] = df['true_range'] / df['true_range'].shift(8) - 1
    
    # Price-Volume Convergence
    df['price_volume_momentum_diff'] = df['price_momentum_5'] - df['volume_momentum_5']
    df['price_volume_accel_diff'] = df['price_acceleration'] - df['volume_acceleration']
    df['price_volume_conv_div'] = df['price_volume_momentum_diff'] - df['price_volume_momentum_diff'].shift(1)
    
    # Price-Range Convergence
    df['price_range_momentum_diff'] = df['price_momentum_5'] - df['range_momentum_5']
    df['price_efficiency'] = (df['close'] / df['close'].shift(1) - 1) / df['true_range']
    df['range_compression'] = df['true_range'] / df['true_range'].rolling(window=5).mean()
    
    # Volume-Range Convergence
    df['volume_range_momentum_diff'] = df['volume_momentum_5'] - df['range_momentum_5']
    df['volume_intensity'] = df['volume'] / df['true_range']
    df['range_volume_efficiency'] = df['true_range'] / df['volume']
    
    # Multi-timeframe convergence strength
    df['short_term_conv'] = (
        (df['price_momentum_3'] * df['volume_momentum_3'] > 0).astype(int) +
        (df['price_momentum_3'] * df['range_momentum_3'] > 0).astype(int) +
        (df['volume_momentum_3'] * df['range_momentum_3'] > 0).astype(int)
    )
    
    df['medium_term_conv'] = (
        (df['price_momentum_8'] * df['volume_momentum_8'] > 0).astype(int) +
        (df['price_momentum_8'] * df['range_momentum_8'] > 0).astype(int) +
        (df['volume_momentum_8'] * df['range_momentum_8'] > 0).astype(int)
    )
    
    df['conv_persistence'] = df['short_term_conv'].rolling(window=3).mean() + df['medium_term_conv'].rolling(window=3).mean()
    
    # Dimension effectiveness (rolling correlations)
    df['price_effectiveness'] = df['price_momentum_5'].rolling(window=10).corr(df['close'].pct_change().shift(-1).rolling(window=5).mean())
    df['volume_effectiveness'] = df['volume_momentum_5'].rolling(window=10).corr(df['close'].pct_change().shift(-1).rolling(window=5).mean())
    df['range_effectiveness'] = df['range_momentum_5'].rolling(window=10).corr(df['close'].pct_change().shift(-1).rolling(window=5).mean())
    
    # Dynamic weights (avoiding division by zero)
    effectiveness_sum = df['price_effectiveness'].abs() + df['volume_effectiveness'].abs() + df['range_effectiveness'].abs()
    df['price_weight'] = df['price_effectiveness'].abs() / np.where(effectiveness_sum == 0, 1, effectiveness_sum)
    df['volume_weight'] = df['volume_effectiveness'].abs() / np.where(effectiveness_sum == 0, 1, effectiveness_sum)
    df['range_weight'] = df['range_effectiveness'].abs() / np.where(effectiveness_sum == 0, 1, effectiveness_sum)
    
    # Convergence acceleration
    df['conv_momentum'] = df['conv_persistence'].diff(3)
    
    # Market regime detection
    df['volatility_ratio'] = df['true_range'].rolling(window=5).std() / df['true_range'].rolling(window=20).std()
    df['trend_strength'] = df['close'].rolling(window=10).apply(lambda x: abs(x[-1] - x[0]) / (x.std() + 1e-8))
    df['market_regime'] = np.where(
        df['trend_strength'] > 1.5, 'trending',
        np.where(df['volatility_ratio'] > 1.2, 'high_vol', 'mean_revert')
    )
    
    # Regime-specific convergence factors
    df['regime_conv_factor'] = np.where(
        df['market_regime'] == 'trending',
        df['conv_persistence'] * 1.5,
        np.where(
            df['market_regime'] == 'high_vol',
            df['conv_persistence'] * 2.0,
            df['conv_persistence'] * 0.7
        )
    )
    
    # Final alpha factor construction
    df['cross_dim_conv'] = (
        df['price_weight'] * df['price_volume_momentum_diff'] +
        df['volume_weight'] * df['volume_range_momentum_diff'] +
        df['range_weight'] * df['price_range_momentum_diff']
    )
    
    df['adaptive_conv_score'] = (
        df['cross_dim_conv'] * df['regime_conv_factor'] * 
        (1 + df['conv_momentum']) * df['conv_persistence']
    )
    
    # Final alpha factor with regime adaptation
    alpha = (
        df['adaptive_conv_score'] * 
        df['price_efficiency'].fillna(0) * 
        (1 + df['volume_intensity'].fillna(0).rolling(window=5).mean())
    )
    
    return alpha
