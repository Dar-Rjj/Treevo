import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Liquidity Momentum with Regime-Adaptive Efficiency
    """
    data = df.copy()
    
    # Multi-Scale Liquidity Efficiency Framework
    # Fractal Range Utilization with Volume Context
    for window in [1, 3, 5]:
        data[f'range_efficiency_{window}d'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
        data[f'volume_weight_{window}d'] = data['volume'] / data['amount']
    
    # Volume-weighted efficiency persistence
    data['efficiency_persistence_3d'] = (data['range_efficiency_1d'].rolling(3).mean() * 
                                        data['volume_weight_1d'].rolling(3).mean())
    data['efficiency_persistence_5d'] = (data['range_efficiency_3d'].rolling(5).mean() * 
                                        data['volume_weight_3d'].rolling(5).mean())
    
    # Microstructure Pressure with Liquidity Classification
    data['daily_range'] = data['high'] - data['low']
    data['range_volume_ratio'] = data['daily_range'] / data['volume'].replace(0, np.nan)
    data['directional_pressure'] = (data['close'] - data['low']) / data['daily_range'].replace(0, np.nan)
    
    # Liquidity state classification
    data['liquidity_state'] = np.where(data['range_volume_ratio'] < data['range_volume_ratio'].rolling(10).median(), 
                                      'compressed', 'expanded')
    
    # Pressure persistence across regimes
    compressed_mask = data['liquidity_state'] == 'compressed'
    expanded_mask = data['liquidity_state'] == 'expanded'
    
    data['pressure_compressed'] = data['directional_pressure'].where(compressed_mask).rolling(5).mean()
    data['pressure_expanded'] = data['directional_pressure'].where(expanded_mask).rolling(5).mean()
    
    # Smart Flow Integration with Range Compression
    # Capital Flow Concentration Analysis
    data['flow_intensity'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['flow_asymmetry'] = (data['flow_intensity'].rolling(5).apply(lambda x: (x - x.mean()).sum()) / 
                             data['flow_intensity'].rolling(5).std().replace(0, np.nan))
    
    # Range compression analysis
    data['range_compression_5d'] = data['daily_range'] / data['daily_range'].rolling(5).mean()
    data['flow_efficiency'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Compression-flow alignment
    data['compression_flow_alignment'] = (data['range_compression_5d'] * data['flow_efficiency']).rolling(3).mean()
    
    # Fractal Momentum with Liquidity Anchoring
    # Multi-Scale Momentum Efficiency
    for window in [1, 3, 5]:
        data[f'momentum_{window}d'] = data['close'].pct_change(window)
    
    # Momentum persistence across liquidity states
    data['momentum_persistence_compressed'] = data['momentum_3d'].where(compressed_mask).rolling(5).std()
    data['momentum_persistence_expanded'] = data['momentum_3d'].where(expanded_mask).rolling(5).std()
    
    # Regime-adaptive efficiency signals
    # High efficiency + strong pressure + liquidity compression
    quality_momentum = ((data['efficiency_persistence_3d'] > data['efficiency_persistence_3d'].rolling(10).quantile(0.7)) &
                       (data['pressure_compressed'] > data['pressure_compressed'].rolling(10).quantile(0.6)) &
                       (data['range_compression_5d'] < 0.8))
    
    # Low efficiency + weak pressure + liquidity expansion
    reversal_confirmation = ((data['efficiency_persistence_3d'] < data['efficiency_persistence_3d'].rolling(10).quantile(0.3)) &
                           (data['pressure_expanded'] < data['pressure_expanded'].rolling(10).quantile(0.4)) &
                           (data['range_compression_5d'] > 1.2))
    
    # Flow-enhanced compression signals
    breakout_conviction = ((data['range_compression_5d'] < 0.7) &
                          (data['flow_asymmetry'].abs() > data['flow_asymmetry'].abs().rolling(10).quantile(0.7)) &
                          (data['compression_flow_alignment'] > data['compression_flow_alignment'].rolling(10).quantile(0.6)))
    
    # Anchored momentum integration
    trend_acceleration = ((data['momentum_3d'] > data['momentum_3d'].rolling(10).quantile(0.7)) &
                         (data['efficiency_persistence_3d'] > data['efficiency_persistence_3d'].rolling(10).quantile(0.6)) &
                         (data['flow_asymmetry'].abs() > data['flow_asymmetry'].abs().rolling(10).quantile(0.6)))
    
    # Composite fractal liquidity momentum factor
    factor = (
        quality_momentum.astype(float) * 0.3 +
        (~reversal_confirmation).astype(float) * 0.2 +
        breakout_conviction.astype(float) * 0.25 +
        trend_acceleration.astype(float) * 0.25 +
        data['efficiency_persistence_5d'].fillna(0) * 0.1 +
        data['compression_flow_alignment'].fillna(0) * 0.1 +
        data['momentum_persistence_compressed'].fillna(0) * 0.05
    )
    
    # Normalize and smooth the factor
    factor = factor.rolling(3).mean()
    factor = (factor - factor.rolling(20).mean()) / factor.rolling(20).std().replace(0, np.nan)
    
    return factor
