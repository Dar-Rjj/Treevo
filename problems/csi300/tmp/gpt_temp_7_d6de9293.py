import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize epsilon for numerical stability
    epsilon = 1e-8
    
    # Fractal Price Structure Analysis
    # Multi-scale Fractal Dimension
    data['short_term_fractal'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + epsilon)
    data['medium_term_fractal'] = (data['high'] - data['low']) / (data['high'].shift(3) - data['low'].shift(3) + epsilon)
    data['fractal_divergence'] = np.abs(data['short_term_fractal'] - data['medium_term_fractal'])
    
    # Price Level Fracture Detection
    # Support/Resistance Proximity
    data['resistance_distance'] = data['high'] / data['high'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else 1.0)
    data['support_distance'] = data['low'] / data['low'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].min() if len(x) > 1 else 1.0)
    
    # Level Breakthrough Signal
    data['resistance_break'] = (data['high'] > data['high'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else -np.inf)).astype(int)
    data['support_break'] = (data['low'] < data['low'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].min() if len(x) > 1 else np.inf)).astype(int)
    
    # Asymmetric Volume Dynamics
    # Volume Distribution Skewness
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['up_volume'] = data['volume'] * (data['price_change'] > 0).astype(float)
    data['down_volume'] = data['volume'] * (data['price_change'] < 0).astype(float)
    data['volume_skew_ratio'] = (data['up_volume'] - data['down_volume']) / (data['up_volume'] + data['down_volume'] + epsilon)
    
    # Volume Persistence Patterns
    # Volume Cluster Detection
    volume_increase = data['volume'] > data['volume'].shift(1)
    streak = volume_increase.groupby((~volume_increase).cumsum()).cumcount() + 1
    data['volume_streak'] = streak * volume_increase
    
    # Asymmetric Volume Momentum
    data['volume_acceleration'] = (data['volume'] - data['volume'].shift(1)) / (data['volume'].shift(1) - data['volume'].shift(2) + epsilon)
    data['directional_volume_momentum'] = data['volume_acceleration'] * np.sign(data['price_change'])
    
    # Price-Volume Fractal Synchronization
    # Fractal Volume Efficiency
    data['volume_per_price_unit'] = data['volume'] / (data['high'] - data['low'] + epsilon)
    data['fractal_volume_ratio'] = data['volume_per_price_unit'] / data['volume_per_price_unit'].rolling(window=5, min_periods=1).mean()
    
    # Multi-scale Synchronization
    data['short_term_sync'] = np.sign(data['price_change']) * np.sign(data['volume'] - data['volume'].shift(1))
    data['medium_term_sync'] = np.sign(data['close'] - data['close'].shift(3)) * np.sign(data['volume'] - data['volume'].shift(3))
    data['sync_divergence'] = (data['short_term_sync'] != data['medium_term_sync']).astype(int)
    
    # Dynamic Range Asymmetry
    # Intraday Range Distribution
    data['upper_range_fraction'] = (data['high'] - data['open']) / (data['high'] - data['low'] + epsilon)
    data['lower_range_fraction'] = (data['open'] - data['low']) / (data['high'] - data['low'] + epsilon)
    
    # Range Asymmetry Patterns
    data['upper_dominance'] = (data['upper_range_fraction'] > 0.7).astype(int)
    data['lower_dominance'] = (data['lower_range_fraction'] > 0.7).astype(int)
    data['balanced_range'] = (np.abs(data['upper_range_fraction'] - 0.5) < 0.2).astype(int)
    
    # Fractal Momentum Breakpoints
    # Multi-timeframe Momentum Fracture
    data['short_term_momentum'] = data['price_change'] / (data['high'].shift(1) - data['low'].shift(1) + epsilon)
    
    def rolling_range_avg(series):
        return (series.rolling(window=4, min_periods=1).apply(lambda x: (x.max() - x.min()) if len(x) > 0 else 1.0))
    
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(3)) / rolling_range_avg(data['high'] - data['low'])
    data['momentum_gap'] = np.abs(data['short_term_momentum'] - data['medium_term_momentum'])
    
    # Volume-Confirmed Breakpoints
    data['high_volume_break'] = ((data['momentum_gap'] > 0.1) & (data['volume_skew_ratio'] > 0.3)).astype(int)
    data['low_volume_rejection'] = ((data['momentum_gap'] > 0.1) & (data['volume_skew_ratio'] < -0.3)).astype(int)
    
    # Alpha Signal Integration
    # Fractal Breakthrough Signals
    data['volume_confirmed_resistance_break'] = (data['resistance_break'] & (data['volume_skew_ratio'] > 0.4)).astype(int)
    data['support_break_volume_cluster'] = (data['support_break'] & (data['volume_streak'] > 2)).astype(int)
    
    # Asymmetric Range Momentum
    data['upper_dominance_momentum'] = data['upper_dominance'] * data['short_term_momentum']
    data['lower_recovery_signal'] = (data['lower_dominance'] & (data['volume_acceleration'] > 0)).astype(int)
    
    # Fractal Synchronization Alpha
    data['positive_sync_divergence'] = (data['sync_divergence'] & (data['volume_skew_ratio'] > 0.2)).astype(int)
    data['fractal_volume_efficiency_signal'] = ((data['fractal_volume_ratio'] > 1.5) & (data['momentum_gap'] > 0.15)).astype(int)
    
    # Final Alpha Factor Construction
    alpha = (
        data['volume_confirmed_resistance_break'] * 0.15 +
        data['support_break_volume_cluster'] * 0.12 +
        data['upper_dominance_momentum'] * 0.10 +
        data['lower_recovery_signal'] * 0.08 +
        data['positive_sync_divergence'] * 0.10 +
        data['fractal_volume_efficiency_signal'] * 0.12 +
        data['high_volume_break'] * 0.08 -
        data['low_volume_rejection'] * 0.10 +
        data['volume_skew_ratio'] * 0.15
    )
    
    return alpha
