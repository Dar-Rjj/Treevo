import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility-Contextualized Price Acceleration
    # Multi-Timeframe Velocity Calculation
    data['short_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['medium_momentum'] = data['close'] / data['close'].shift(10) - 1
    data['acceleration'] = data['short_momentum'] - data['medium_momentum']
    
    # Volatility Scaling Framework
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    data['volatility_20d'] = data['high_low_range'].rolling(window=20, min_periods=10).std()
    data['scaled_acceleration'] = data['acceleration'] / data['volatility_20d'].shift(1)
    
    # Acceleration persistence (count of consecutive same-sign values)
    data['accel_sign'] = np.sign(data['scaled_acceleration'])
    data['accel_sign_change'] = data['accel_sign'] != data['accel_sign'].shift(1)
    data['accel_persistence'] = data.groupby(data['accel_sign_change'].cumsum()).cumcount() + 1
    
    # Acceleration Quality Assessment
    data['accel_magnitude_ratio'] = abs(data['scaled_acceleration']) / data['volatility_20d'].shift(1)
    data['accel_consistency_3d'] = data['scaled_acceleration'].rolling(window=3, min_periods=2).std()
    data['accel_consistency_5d'] = data['scaled_acceleration'].rolling(window=5, min_periods=3).std()
    data['accel_quality'] = data['accel_magnitude_ratio'] / (data['accel_consistency_3d'] + data['accel_consistency_5d'] + 1e-8)
    
    # Volume-Price Divergence with Breakthrough Detection
    # Extreme Price Level Analysis
    data['highest_10d'] = data['high'].rolling(window=10, min_periods=5).max()
    data['lowest_10d'] = data['low'].rolling(window=10, min_periods=5).min()
    data['breakthrough_up'] = (data['close'] > data['highest_10d'].shift(1)).astype(int)
    data['breakthrough_down'] = (data['close'] < data['lowest_10d'].shift(1)).astype(int)
    
    # Volume-Price Divergence Quantification
    data['volume_rank'] = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['price_rank'] = data['close'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['divergence_magnitude'] = abs(data['volume_rank'] - data['price_rank'])
    
    # Volatility Expansion Confirmation
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['avg_true_range_10d'] = data['true_range'].rolling(window=10, min_periods=5).mean()
    data['volatility_expansion'] = data['true_range'] / data['avg_true_range_10d'].shift(1)
    
    # Acceleration-Divergence Convergence Analysis
    # Signal Alignment Detection
    data['accel_divergence_alignment'] = (
        (data['scaled_acceleration'] > 0) & (data['breakthrough_up'] == 1) |
        (data['scaled_acceleration'] < 0) & (data['breakthrough_down'] == 1)
    ).astype(int)
    
    # Convergence Strength Assessment
    data['convergence_magnitude'] = (
        abs(data['scaled_acceleration']) * data['divergence_magnitude'] * data['accel_persistence']
    )
    data['signal_noise_ratio'] = data['convergence_magnitude'] / (data['volatility_20d'].shift(1) + 1e-8)
    
    # Volume Confirmation Enhancement
    data['avg_volume_10d'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_confirmation'] = data['volume'] / data['avg_volume_10d'].shift(1)
    data['breakthrough_volume_weight'] = np.where(
        (data['breakthrough_up'] == 1) | (data['breakthrough_down'] == 1),
        data['volume_confirmation'],
        1.0
    )
    
    # Integrated Alpha Factor Synthesis
    # Base Signal Construction
    volatility_scaled_acceleration = data['scaled_acceleration'] * data['accel_quality']
    breakthrough_weighted_divergence = (
        data['divergence_magnitude'] * 
        (data['breakthrough_up'] - data['breakthrough_down']) * 
        data['volatility_expansion']
    )
    
    # Signal Quality Enhancement
    persistence_weighted_accel = volatility_scaled_acceleration * data['accel_persistence']
    volatility_expansion_confirmed = breakthrough_weighted_divergence * data['volatility_expansion']
    volume_confirmed_signals = volatility_expansion_confirmed * data['breakthrough_volume_weight']
    
    # Final Factor Output
    convergence_strength = data['convergence_magnitude'] * data['signal_noise_ratio']
    aligned_signals = data['accel_divergence_alignment'] * convergence_strength
    
    final_factor = (
        persistence_weighted_accel * 
        volume_confirmed_signals * 
        aligned_signals * 
        data['accel_quality']
    )
    
    # Clean up and return
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    return final_factor
