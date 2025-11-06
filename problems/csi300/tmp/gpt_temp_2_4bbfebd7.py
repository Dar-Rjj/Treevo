import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining price-volume regime momentum, range efficiency persistence,
    amount flow direction quality, volume-confirmed extreme reversal, and regime-adaptive volume clustering.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Regime Momentum
    # Volatility-Normalized Price Momentum
    data['short_momentum'] = (data['close'] / data['close'].shift(5) - 1) / data['high'].rolling(6).apply(lambda x: (x - data['low'].reindex(x.index)).std(), raw=False)
    data['medium_momentum'] = (data['close'] / data['close'].shift(10) - 1) / data['high'].rolling(11).apply(lambda x: (x - data['low'].reindex(x.index)).std(), raw=False)
    data['momentum_persistence'] = ((data['close'] / data['close'].shift(5) - 1) * (data['close'] / data['close'].shift(10) - 1) > 0).astype(int)
    
    # Volume Persistence Regime
    data['volume_trend_strength'] = (data['volume'] / data['volume'].shift(5)) - (data['volume'].shift(5) / data['volume'].shift(10))
    data['volume_regime'] = ((data['volume'] > data['volume'].shift(1)) & (data['volume'].shift(1) > data['volume'].shift(2))).astype(int)
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(3)) / (data['volume'].shift(3) / data['volume'].shift(6))
    
    # Divergence Blend
    data['clean_divergence'] = data['short_momentum'] - data['volume_acceleration']
    data['confirmation'] = data['medium_momentum'] + data['volume_regime']
    data['regime_weighted_signal'] = data['clean_divergence'] * data['volume_regime'].rolling(3).mean()
    
    # Range Efficiency Persistence
    # Multi-timeframe Efficiency
    data['efficiency_3d'] = abs(data['close'] - data['close'].shift(3)) / data['high'].rolling(4).apply(lambda x: (x - data['low'].reindex(x.index)).sum(), raw=False)
    data['efficiency_5d'] = abs(data['close'] - data['close'].shift(5)) / data['high'].rolling(6).apply(lambda x: (x - data['low'].reindex(x.index)).sum(), raw=False)
    data['efficiency_trend'] = data['efficiency_5d'] / data['efficiency_3d']
    
    # Gap-Adjusted True Range
    data['adjusted_range'] = np.maximum(data['high'], data['close'].shift(1)) - np.minimum(data['low'], data['close'].shift(1))
    data['gap_efficiency'] = abs(data['close'] - data['close'].shift(1)) / data['adjusted_range']
    
    # Efficiency Regime
    efficiency_median = data['efficiency_5d'].rolling(20).median()
    data['high_efficiency_persistence'] = (data['efficiency_5d'] > efficiency_median).rolling(5).sum()
    data['efficiency_breakdown'] = data['adjusted_range'] - abs(data['close'] - data['close'].shift(1))
    
    # Amount Flow Direction Quality
    # Pure Flow Direction
    up_flow = data['amount'].where(data['close'] > data['close'].shift(1), 0)
    down_flow = data['amount'].where(data['close'] < data['close'].shift(1), 0)
    data['flow_ratio'] = up_flow.rolling(5).sum() / (up_flow.rolling(5).sum() + down_flow.rolling(5).sum())
    
    # Flow Persistence Pattern
    price_diff_sign = np.sign(data['close'] - data['close'].shift(1))
    data['consecutive_directional_flow'] = (price_diff_sign == price_diff_sign.shift(1)).rolling(3).sum()
    data['flow_acceleration'] = (data['amount'] / data['amount'].shift(3)) / (data['amount'].shift(3) / data['amount'].shift(6))
    data['flow_quality'] = data['flow_ratio'] * data['consecutive_directional_flow']
    
    # Volume-Confirmed Extreme Reversal
    # Volatility-Scaled Extremes
    rolling_std = data['high'].rolling(6).apply(lambda x: (x - data['low'].reindex(x.index)).std(), raw=False)
    data['price_extremity'] = (data['close'] - data['close'].shift(1)) / rolling_std
    data['volume_extremity'] = data['volume'] / data['volume'].rolling(6).median()
    data['combined_extreme'] = data['price_extremity'] * data['volume_extremity']
    
    # Multi-day Confirmation
    data['pre_extreme_trend'] = np.sign(data['close'].shift(1) - data['close'].shift(3))
    data['post_extreme_behavior'] = np.sign(data['close'] - data['close'].shift(1))
    data['volume_persistence_after_extreme'] = (data['volume'] > data['volume'].shift(1)).rolling(3).sum()
    
    # Regime-Adaptive Volume Clustering
    # Pure Volatility Regime
    data['range_volatility'] = (data['high'].rolling(6).max() - data['low'].rolling(6).min()) / data['close'].shift(5)
    data['volatility_trend'] = data['range_volatility'] / data['range_volatility'].shift(5)
    
    # Volume Cluster Patterns
    volume_median = data['volume'].rolling(20).median()
    data['volume_spike_clustering'] = (data['volume'] > 2 * volume_median).rolling(5).sum()
    data['volume_persistence_streak'] = (data['volume'] > data['volume'].shift(1)).rolling(5).sum()
    
    # Final alpha factor combining all components
    alpha = (
        0.2 * data['regime_weighted_signal'] +
        0.15 * data['efficiency_trend'] +
        0.15 * data['flow_quality'] +
        0.25 * data['combined_extreme'] * data['post_extreme_behavior'] +
        0.25 * np.where(data['range_volatility'] > data['range_volatility'].rolling(20).median(),
                       data['volume_spike_clustering'],
                       -data['volume_spike_clustering'])
    )
    
    return alpha
