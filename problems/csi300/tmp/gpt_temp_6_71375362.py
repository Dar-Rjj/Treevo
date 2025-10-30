import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Efficiency Breakout with Volume-Weighted Momentum Divergence
    """
    data = df.copy()
    
    # Multi-Timeframe Efficiency Breakout Detection
    # Volatility-Adjusted Breakout Identification
    data['breakout_3d'] = data['close'] / data['high'].rolling(window=3, min_periods=3).max() - 1
    data['breakout_8d'] = data['close'] / data['high'].rolling(window=8, min_periods=8).max() - 1
    data['breakout_21d'] = data['close'] / data['high'].rolling(window=21, min_periods=21).max() - 1
    
    # Fractal Efficiency Assessment
    data['hl_range_3d'] = (data['high'] - data['low']).rolling(window=3, min_periods=3).sum()
    data['hl_range_8d'] = (data['high'] - data['low']).rolling(window=8, min_periods=8).sum()
    data['hl_range_21d'] = (data['high'] - data['low']).rolling(window=21, min_periods=21).sum()
    
    data['efficiency_3d'] = np.abs(data['close'] - data['close'].shift(3)) / data['hl_range_3d']
    data['efficiency_8d'] = np.abs(data['close'] - data['close'].shift(8)) / data['hl_range_8d']
    data['efficiency_21d'] = np.abs(data['close'] - data['close'].shift(21)) / data['hl_range_21d']
    
    # Efficiency-Breakout Alignment Analysis
    data['efficiency_alignment'] = (
        np.sign(data['breakout_3d']) * data['efficiency_3d'] +
        np.sign(data['breakout_8d']) * data['efficiency_8d'] +
        np.sign(data['breakout_21d']) * data['efficiency_21d']
    ) / 3
    
    # Volume-Weighted Momentum Divergence Component
    # Multi-Timeframe Volume-Weighted Momentum
    def volume_weighted_momentum(window):
        vol_momentum = []
        for i in range(len(data)):
            if i < window:
                vol_momentum.append(np.nan)
                continue
            window_data = data.iloc[i-window:i+1]
            numerator = (window_data['volume'] * (window_data['close'] - window_data['close'].shift(1))).sum()
            denominator = np.abs(window_data['close'] - window_data['close'].shift(1)).sum()
            vol_momentum.append(numerator / denominator if denominator != 0 else 0)
        return vol_momentum
    
    data['vw_momentum_3d'] = volume_weighted_momentum(3)
    data['vw_momentum_8d'] = volume_weighted_momentum(8)
    data['vw_momentum_21d'] = volume_weighted_momentum(21)
    
    # Range-Normalized Momentum
    data['range_norm_momentum_3d'] = (
        (data['close'] - data['close'].shift(3)) / 
        (data['high'].rolling(window=3, min_periods=3).max() - data['low'].rolling(window=3, min_periods=3).min())
    )
    data['range_norm_momentum_8d'] = (
        (data['close'] - data['close'].shift(8)) / 
        (data['high'].rolling(window=8, min_periods=8).max() - data['low'].rolling(window=8, min_periods=8).min())
    )
    data['range_norm_momentum_21d'] = (
        (data['close'] - data['close'].shift(21)) / 
        (data['high'].rolling(window=21, min_periods=21).max() - data['low'].rolling(window=21, min_periods=21).min())
    )
    
    # Momentum Divergence Detection
    data['momentum_divergence_3d'] = data['vw_momentum_3d'] - data['range_norm_momentum_3d']
    data['momentum_divergence_8d'] = data['vw_momentum_8d'] - data['range_norm_momentum_8d']
    data['momentum_divergence_21d'] = data['vw_momentum_21d'] - data['range_norm_momentum_21d']
    
    # Volume-Enhanced Regime Filtering
    # Volume Profile Analysis
    data['volume_trend_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_trend_20d'] = data['volume'] / data['volume'].shift(20) - 1
    
    data['volume_volatility'] = (
        data['volume'].rolling(window=10, min_periods=10).std() / 
        data['volume'].rolling(window=10, min_periods=10).mean()
    )
    
    # Efficiency-Breakout-Volume Regime Detection
    data['breakout_strength'] = (
        data['breakout_3d'] * data['efficiency_3d'] +
        data['breakout_8d'] * data['efficiency_8d'] +
        data['breakout_21d'] * data['efficiency_21d']
    ) / 3
    
    data['momentum_alignment'] = (
        np.sign(data['vw_momentum_3d']) * np.abs(data['vw_momentum_3d']) +
        np.sign(data['vw_momentum_8d']) * np.abs(data['vw_momentum_8d']) +
        np.sign(data['vw_momentum_21d']) * np.abs(data['vw_momentum_21d'])
    ) / 3
    
    # Adaptive Signal Integration Framework
    # Multi-Dimensional Confidence Scoring
    data['efficiency_confidence'] = (
        data['efficiency_3d'].rolling(window=5, min_periods=5).std() +
        data['efficiency_8d'].rolling(window=5, min_periods=5).std() +
        data['efficiency_21d'].rolling(window=5, min_periods=5).std()
    ) / 3
    
    data['momentum_confidence'] = (
        np.abs(data['momentum_divergence_3d']).rolling(window=5, min_periods=5).std() +
        np.abs(data['momentum_divergence_8d']).rolling(window=5, min_periods=5).std() +
        np.abs(data['momentum_divergence_21d']).rolling(window=5, min_periods=5).std()
    ) / 3
    
    # Dynamic Component Weighting
    efficiency_weight = 1 / (1 + data['efficiency_confidence'])
    momentum_weight = 1 / (1 + data['momentum_confidence'])
    volume_weight = 1 / (1 + data['volume_volatility'])
    
    # Composite Alpha Factor Construction
    # Core Signal Generation
    efficiency_component = (
        data['breakout_3d'] * data['efficiency_3d'] * efficiency_weight +
        data['breakout_8d'] * data['efficiency_8d'] * efficiency_weight +
        data['breakout_21d'] * data['efficiency_21d'] * efficiency_weight
    ) / 3
    
    momentum_component = (
        data['momentum_divergence_3d'] * momentum_weight +
        data['momentum_divergence_8d'] * momentum_weight +
        data['momentum_divergence_21d'] * momentum_weight
    ) / 3
    
    volume_component = (
        data['volume_trend_5d'] * volume_weight +
        data['volume_trend_20d'] * volume_weight
    ) / 2
    
    # Final Alpha Output
    alpha = (
        efficiency_component * 0.4 +
        momentum_component * 0.4 +
        volume_component * 0.2
    )
    
    # Signal Refinement
    alpha = alpha.rolling(window=3, min_periods=3).mean()  # Smoothing
    alpha = alpha / alpha.rolling(window=21, min_periods=21).std()  # Normalization
    
    return alpha
