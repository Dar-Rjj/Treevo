import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Divergence Framework
    df = df.copy()
    
    # Raw Divergence Signals
    df['price_momentum'] = df['close'] / df['close'].shift(1) - 1
    df['volume_momentum'] = df['volume'] / df['volume'].shift(1) - 1
    df['raw_divergence'] = df['price_momentum'] - df['volume_momentum']
    
    # Divergence Persistence
    df['div_sign'] = np.sign(df['raw_divergence'])
    df['direction_persistence'] = df['div_sign'].rolling(window=3, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) == 3 else np.nan, raw=False
    )
    df['magnitude_persistence'] = df['raw_divergence'] / df['raw_divergence'].abs().rolling(window=5, min_periods=1).mean()
    
    # Divergence Acceleration
    df['divergence_acceleration'] = df['raw_divergence'] - df['raw_divergence'].rolling(window=3, min_periods=1).mean()
    
    # Volatility-Adjusted Components
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['close_volatility'] = abs(df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['range_adjusted_divergence'] = df['raw_divergence'] / (df['daily_range'] + 1e-8)
    df['volatility_stable_divergence'] = df['raw_divergence'] * (df['close_volatility'] / (df['daily_range'] + 1e-8))
    
    # Asymmetry Detection
    df['positive_strength'] = np.where(df['raw_divergence'] > 0, df['raw_divergence'], 0)
    df['negative_strength'] = np.where(df['raw_divergence'] < 0, df['raw_divergence'], 0)
    df['asymmetry_ratio'] = df['positive_strength'] / (abs(df['negative_strength']) + 0.001)
    
    df['large_divergence'] = (df['raw_divergence'].abs() > 
                             df['raw_divergence'].abs().rolling(window=10, min_periods=1).mean()).astype(int)
    
    def count_asymmetry(x):
        if len(x) < 5:
            return np.nan
        return (x > 1).sum() - (x < 1).sum()
    
    df['asymmetry_persistence'] = df['asymmetry_ratio'].rolling(window=5, min_periods=1).apply(
        count_asymmetry, raw=False
    )
    
    # Multi-Timeframe Dynamics
    df['short_term_momentum'] = df['raw_divergence'] / (df['raw_divergence'].rolling(window=2, min_periods=1).mean() + 1e-8) - 1
    df['short_term_asymmetry'] = df['asymmetry_ratio'] / (df['asymmetry_ratio'].rolling(window=3, min_periods=1).mean() + 1e-8)
    df['medium_term_trend'] = df['raw_divergence'].rolling(window=5, min_periods=1).mean()
    df['medium_term_stability'] = df['raw_divergence'].rolling(window=10, min_periods=1).std()
    
    # Volume-Intensity Enhancement
    df['volume_intensity'] = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
    df['high_intensity_divergence'] = df['raw_divergence'] * df['volume_intensity']
    
    df['trade_size'] = df['amount'] / (df['volume'] + 1e-8)
    df['size_divergence_alignment'] = df['raw_divergence'] * (df['trade_size'] / df['trade_size'].rolling(window=5, min_periods=1).mean() - 1)
    
    # Price-Level Adaptation
    df['price_context_momentum'] = df['close'] / df['close'].rolling(window=10, min_periods=1).mean() - 1
    df['level_adjusted_divergence'] = df['raw_divergence'] / (df['price_context_momentum'] + 1e-8)
    
    df['daily_position'] = (df['close'] - df['low']) / ((df['high'] - df['low']) + 1e-8)
    df['position_divergence'] = df['raw_divergence'] * (df['daily_position'] - df['daily_position'].rolling(window=5, min_periods=1).mean())
    
    # Regime Classification
    div_std_20 = df['raw_divergence'].rolling(window=20, min_periods=1).std()
    df['high_regime'] = (df['raw_divergence'].abs() > div_std_20).astype(int)
    df['low_regime'] = (df['raw_divergence'].abs() < 0.5 * div_std_20).astype(int)
    
    df['high_regime_factor'] = df['range_adjusted_divergence'] * df['daily_range']
    df['low_regime_factor'] = df['raw_divergence'].rolling(window=5, min_periods=1).sum()
    df['regime_change'] = df['raw_divergence'] * (df['high_regime'] != df['high_regime'].shift(1)).astype(int)
    
    # Composite Alpha Factor
    # Core components
    core_divergence = df['volatility_stable_divergence'].fillna(0)
    core_asymmetry = df['asymmetry_persistence'].fillna(0)
    core_multi_timeframe = (df['short_term_momentum'].fillna(0) + 
                           df['medium_term_trend'].fillna(0) - 
                           df['medium_term_stability'].fillna(0))
    
    # Enhancement layers
    volume_enhancement = df['high_intensity_divergence'].fillna(0) + df['size_divergence_alignment'].fillna(0)
    price_enhancement = df['level_adjusted_divergence'].fillna(0) + df['position_divergence'].fillna(0)
    regime_enhancement = (df['high_regime_factor'].fillna(0) + 
                         df['low_regime_factor'].fillna(0) + 
                         df['regime_change'].fillna(0))
    
    # Adaptive weighting
    volatility_weight = 1 / (df['daily_range'].fillna(0.01) + 0.01)
    asymmetry_weight = df['asymmetry_ratio'].fillna(1).abs()
    volume_weight = df['volume_intensity'].fillna(1)
    
    # Final composite factor
    composite_factor = (
        (core_divergence * volatility_weight) +
        (core_asymmetry * asymmetry_weight) +
        (core_multi_timeframe) +
        (volume_enhancement * volume_weight) +
        price_enhancement +
        regime_enhancement
    )
    
    return composite_factor
