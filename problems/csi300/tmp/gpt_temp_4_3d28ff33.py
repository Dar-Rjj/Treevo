import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Directional Efficiency Framework
    # Raw Efficiency Components
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Multi-period range efficiency (5-day)
    data['close_5d_ago'] = data['close'].shift(5)
    data['range_sum_5d'] = (data['high'] - data['low']).rolling(window=5, min_periods=3).sum()
    data['range_efficiency_5d'] = (data['close'] - data['close_5d_ago']) / data['range_sum_5d']
    
    # Efficiency momentum
    data['efficiency_5d_avg'] = data['intraday_efficiency'].rolling(window=5, min_periods=3).mean()
    data['efficiency_momentum'] = data['intraday_efficiency'] / data['efficiency_5d_avg']
    
    # Efficiency Quality Assessment
    data['efficiency_5d_std'] = data['intraday_efficiency'].rolling(window=5, min_periods=3).std()
    data['efficiency_consistency'] = 1 - (data['efficiency_5d_std'] / data['efficiency_5d_avg'].abs()).replace([np.inf, -np.inf], np.nan)
    
    data['efficiency_10d_avg'] = data['intraday_efficiency'].rolling(window=10, min_periods=5).mean()
    data['efficiency_acceleration'] = (data['efficiency_5d_avg'] - data['efficiency_10d_avg']) / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency_divergence'] = (data['efficiency_5d_avg'] - data['efficiency_10d_avg']).abs()
    
    # Regime-Based Efficiency Classification
    data['efficiency_20d_percentile'] = data['intraday_efficiency'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 10 else np.nan
    )
    data['high_efficiency_regime'] = (data['efficiency_20d_percentile'] > 0.8).astype(int)
    data['low_efficiency_regime'] = (data['efficiency_20d_percentile'] < 0.2).astype(int)
    data['regime_change'] = data['high_efficiency_regime'].diff().abs() + data['low_efficiency_regime'].diff().abs()
    
    # Volume-Price Synchronization Analysis
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_direction_sync'] = np.sign(data['close'] - data['open']) * np.sign(data['volume'] - data['volume_5d_avg'])
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    
    data['amount_5d_avg'] = data['amount'].rolling(window=5, min_periods=3).mean()
    data['volume_amount_quality'] = (data['volume'] / data['volume_5d_avg']) * (data['amount'] / data['amount_5d_avg'])
    
    # Multi-Timeframe Volume Momentum
    data['volume_3d_avg'] = data['volume'].rolling(window=3, min_periods=2).mean()
    data['volume_10d_avg'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_ratio'] = (data['volume'] / data['volume_3d_avg']) / (data['volume'] / data['volume_10d_avg'])
    
    data['volume_weighted_momentum'] = data['intraday_efficiency'] * (data['volume'] / data['volume'].shift(1))
    data['amount_enhanced_momentum'] = data['volume_weighted_momentum'] * (data['amount'] / data['amount_5d_avg'])
    
    # Volume-Efficiency Divergence
    data['efficiency_ratio'] = data['intraday_efficiency'] / data['efficiency_5d_avg']
    data['efficiency_volume_divergence'] = data['efficiency_ratio'] / data['volume_ratio']
    
    # Momentum Hierarchy Framework
    data['return_1d'] = data['close'].pct_change(1)
    data['return_3d'] = data['close'].pct_change(3)
    data['return_5d'] = data['close'].pct_change(5)
    data['return_10d'] = data['close'].pct_change(10)
    data['return_15d'] = data['close'].pct_change(15)
    
    data['momentum_hierarchy'] = (data['return_3d'] / data['return_10d']) * (data['return_5d'] / data['return_15d'])
    data['momentum_stability'] = 1 - (data['return_5d'] / data['return_10d'] - 1).abs()
    
    # Range Breakout Context
    data['high_5d_ago'] = data['high'].rolling(window=5, min_periods=3).max().shift(1)
    data['low_5d_ago'] = data['low'].rolling(window=5, min_periods=3).min().shift(1)
    
    data['resistance_breakthrough'] = (data['close'] - data['high_5d_ago']) / (data['high'] - data['low']).replace(0, np.nan)
    data['support_breakdown'] = (data['low_5d_ago'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Gap Analysis Integration
    data['normalized_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['volume_20d_median'] = data['volume'].rolling(window=20, min_periods=10).median()
    data['volume_adjusted_gap'] = data['normalized_gap'] * (data['volume'] / data['volume_20d_median'])
    data['gap_efficiency'] = (data['close'] - data['open']).abs() / (data['open'] - data['close'].shift(1)).abs().replace(0, np.nan)
    
    # Efficiency-Momentum Divergence Detection
    data['momentum_ratio'] = data['return_3d'] / data['return_10d']
    data['efficiency_momentum_divergence'] = data['efficiency_ratio'] / data['momentum_ratio']
    
    # Adaptive Alpha Signal Generation
    # Core efficiency factor
    data['core_efficiency_factor'] = data['efficiency_acceleration'] * data['volume_weighted_momentum']
    
    # Momentum enhancement
    data['range_compression'] = (data['high'] - data['low']).rolling(window=5, min_periods=3).std()
    data['price_acceleration'] = data['return_1d'].rolling(window=5, min_periods=3).mean()
    data['breakout_efficiency'] = data['price_acceleration'] / data['range_compression'].replace(0, np.nan)
    data['momentum_enhancement'] = data['momentum_hierarchy'] * data['breakout_efficiency']
    
    # Divergence confirmation
    data['divergence_confirmation'] = data['efficiency_momentum_divergence'] * data['volume_direction_sync']
    
    # Regime-adaptive alpha
    regime_weights = (
        data['high_efficiency_regime'] * 0.6 + 
        data['low_efficiency_regime'] * 0.3 + 
        ((1 - data['high_efficiency_regime'] - data['low_efficiency_regime']) * 0.5)
    )
    
    # Composite alpha factor
    alpha_signal = (
        data['core_efficiency_factor'] * 0.4 +
        data['momentum_enhancement'] * 0.3 +
        data['divergence_confirmation'] * 0.3
    ) * regime_weights
    
    # Clean up intermediate columns
    intermediate_cols = [
        'intraday_efficiency', 'close_5d_ago', 'range_sum_5d', 'range_efficiency_5d',
        'efficiency_5d_avg', 'efficiency_momentum', 'efficiency_5d_std', 'efficiency_consistency',
        'efficiency_10d_avg', 'efficiency_acceleration', 'efficiency_divergence',
        'efficiency_20d_percentile', 'high_efficiency_regime', 'low_efficiency_regime',
        'regime_change', 'volume_5d_avg', 'volume_direction_sync', 'volume_efficiency',
        'amount_5d_avg', 'volume_amount_quality', 'volume_3d_avg', 'volume_10d_avg',
        'volume_ratio', 'volume_weighted_momentum', 'amount_enhanced_momentum',
        'efficiency_ratio', 'efficiency_volume_divergence', 'return_1d', 'return_3d',
        'return_5d', 'return_10d', 'return_15d', 'momentum_hierarchy', 'momentum_stability',
        'high_5d_ago', 'low_5d_ago', 'resistance_breakthrough', 'support_breakdown',
        'normalized_gap', 'volume_20d_median', 'volume_adjusted_gap', 'gap_efficiency',
        'momentum_ratio', 'efficiency_momentum_divergence', 'core_efficiency_factor',
        'range_compression', 'price_acceleration', 'breakout_efficiency', 'momentum_enhancement',
        'divergence_confirmation'
    ]
    
    result = alpha_signal.copy()
    result.name = 'alpha_factor'
    
    return result
