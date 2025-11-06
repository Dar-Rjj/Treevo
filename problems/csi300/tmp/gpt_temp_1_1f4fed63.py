import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Anchoring Efficiency with Volume-Regime Synchronization alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price features
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    
    # Multi-Scale Fractal Anchoring
    # Daily Range Anchoring
    df['price_movement_anchoring'] = abs(df['close'] - df['open']) / np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    
    df['volume_anchoring_efficiency'] = df['volume'] / (df['high'] - df['low'] + 1e-8)
    
    # Anchoring Momentum Persistence
    df['close_5d_ago'] = df['close'].shift(5)
    df['volume_1d_ago'] = df['volume'].shift(1)
    df['anchoring_momentum'] = ((df['close'] - df['close_5d_ago']) / (df['close_5d_ago'] + 1e-8)) * \
                              np.sign(df['volume'] / (df['volume_1d_ago'] + 1e-8) - 1)
    
    # Multi-Day Fractal Anchoring
    df['close_10d_ago'] = df['close'].shift(10)
    
    # Calculate rolling price volatility for trend anchoring smoothness
    rolling_volatility = []
    for i in range(len(df)):
        if i >= 10:
            recent_returns = []
            for j in range(i-9, i+1):
                if j > 0:
                    ret = abs(df['close'].iloc[j] - df['close'].iloc[j-1])
                    recent_returns.append(ret)
            if len(recent_returns) > 0 and sum(recent_returns) > 0:
                smoothness = (df['close'].iloc[i] - df['close_10d_ago'].iloc[i]) / (df['close_10d_ago'].iloc[i] + 1e-8) / sum(recent_returns)
            else:
                smoothness = 0
        else:
            smoothness = 0
        rolling_volatility.append(smoothness)
    
    df['trend_anchoring_smoothness'] = rolling_volatility
    
    # Volume-Regime Synchronization Analysis
    df['ultra_short_anchoring'] = (df['close'] - df['close'].shift(3)) / (df['close'].shift(3) + 1e-8)
    df['short_term_anchoring'] = (df['close'] - df['close'].shift(10)) / (df['close'].shift(10) + 1e-8)
    df['medium_term_anchoring'] = (df['close'] - df['close'].shift(21)) / (df['close'].shift(21) + 1e-8)
    
    # Volume Pattern Detection
    df['volume_momentum'] = df['volume'] / (df['volume_1d_ago'] + 1e-8) - 1
    
    # Calculate volume clustering
    volume_clustering = []
    for i in range(len(df)):
        if i >= 5:
            recent_volumes = df['volume'].iloc[i-4:i+1]
            avg_volume = recent_volumes.mean()
            clustering_ratio = df['volume'].iloc[i] / (avg_volume + 1e-8)
        else:
            clustering_ratio = 1
        volume_clustering.append(clustering_ratio)
    
    df['volume_clustering'] = volume_clustering
    
    # Anchoring-Volume Alignment
    df['anchoring_volume_alignment'] = np.sign(df['ultra_short_anchoring']) * np.sign(df['volume_momentum'])
    
    # Asymmetric Volatility Anchoring
    df['ultra_short_volatility'] = df['true_range'].rolling(window=3, min_periods=1).mean()
    df['short_term_volatility'] = df['true_range'].rolling(window=8, min_periods=1).mean()
    df['volatility_momentum'] = df['short_term_volatility'] / (df['ultra_short_volatility'] + 1e-8) - 1
    
    # Price-Level Anchoring Effects
    df['5d_high'] = df['high'].rolling(window=5, min_periods=1).max()
    df['5d_low'] = df['low'].rolling(window=5, min_periods=1).min()
    df['distance_from_5d_high'] = (df['high'] - df['5d_high']) / (df['high'] + 1e-8)
    df['distance_from_5d_low'] = (df['5d_low'] - df['low']) / (df['low'] + 1e-8)
    df['anchoring_strength'] = df['distance_from_5d_high'] - df['distance_from_5d_low']
    
    # Historical Price Memory
    df['21d_high'] = df['high'].rolling(window=21, min_periods=1).max()
    df['21d_low'] = df['low'].rolling(window=21, min_periods=1).min()
    df['21d_high_proximity'] = (df['high'] - df['21d_high']) / (df['high'] + 1e-8)
    df['21d_low_proximity'] = (df['21d_low'] - df['low']) / (df['low'] + 1e-8)
    df['historical_anchor_bias'] = df['21d_high_proximity'] * df['21d_low_proximity']
    
    # Volume-Flow Asymmetry Detection
    up_volume = []
    down_volume = []
    for i in range(len(df)):
        if i > 0:
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                up_volume.append(df['volume'].iloc[i])
                down_volume.append(0)
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                up_volume.append(0)
                down_volume.append(df['volume'].iloc[i])
            else:
                up_volume.append(0)
                down_volume.append(0)
        else:
            up_volume.append(0)
            down_volume.append(0)
    
    df['up_volume'] = up_volume
    df['down_volume'] = down_volume
    df['volume_pressure_asymmetry'] = (df['up_volume'] - df['down_volume']) / (df['up_volume'] + df['down_volume'] + 1e-8)
    
    # Large trade concentration
    df['large_trade_concentration'] = df['amount'] / (df['volume'] + 1e-8)
    
    # Regime-Dependent Signal Construction
    # High volatility regime signals
    df['breakout_anchoring'] = df['distance_from_5d_high'] * df['true_range']
    df['volatility_expansion_factor'] = df['breakout_anchoring'] * df['volatility_momentum']
    df['high_vol_flow_multiplier'] = df['volatility_expansion_factor'] * df['volume_pressure_asymmetry']
    
    # Low volatility regime signals
    df['mean_reversion_pressure'] = df['anchoring_strength'] * df['historical_anchor_bias']
    df['compression_release_factor'] = df['mean_reversion_pressure'] * df['volume_clustering']
    df['low_vol_flow_adjustment'] = df['compression_release_factor'] * df['volume_clustering']
    
    # Composite Alpha Signal Construction
    # Core synchronization signals
    df['strong_sync'] = (df['price_movement_anchoring'] + df['anchoring_volume_alignment'] + 
                         (np.sign(df['ultra_short_anchoring']) == np.sign(df['short_term_anchoring'])).astype(float))
    
    # Multi-dimensional alignment score
    df['multi_dim_alignment'] = (
        df['price_movement_anchoring'] * 
        df['volume_anchoring_efficiency'] * 
        df['anchoring_volume_alignment']
    )
    
    # Final composite signal with regime adjustment
    for i in range(len(df)):
        if i >= 21:  # Ensure we have enough data for calculations
            # Volatility regime detection
            is_high_vol = df['short_term_volatility'].iloc[i] > df['short_term_volatility'].iloc[i-20:i].quantile(0.7)
            
            if is_high_vol:
                regime_signal = df['high_vol_flow_multiplier'].iloc[i]
            else:
                regime_signal = df['low_vol_flow_adjustment'].iloc[i]
            
            # Confidence weighting based on volatility
            volatility_weight = 1.0 / (1.0 + df['short_term_volatility'].iloc[i])
            
            # Final composite signal
            composite_signal = (
                df['multi_dim_alignment'].iloc[i] * 0.3 +
                regime_signal * 0.4 +
                df['strong_sync'].iloc[i] * 0.2 +
                df['trend_anchoring_smoothness'].iloc[i] * 0.1
            ) * volatility_weight
            
            result.iloc[i] = composite_signal
        else:
            result.iloc[i] = 0
    
    # Clean up any infinite or NaN values
    result = result.replace([np.inf, -np.inf], 0).fillna(0)
    
    return result
