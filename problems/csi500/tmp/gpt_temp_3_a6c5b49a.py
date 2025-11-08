import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Calculate price changes
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['abs_price_change'] = df['price_change'].abs()
    
    # Directional Momentum Strength Analysis
    # 4-day momentum components
    df['bullish_momentum'] = df['price_change'].rolling(window=4).apply(
        lambda x: np.sum(np.maximum(x, 0)) / np.sum(np.abs(x)) if np.sum(np.abs(x)) > 0 else 0, raw=True
    )
    
    df['bearish_momentum'] = df['price_change'].rolling(window=4).apply(
        lambda x: np.sum(np.maximum(-x, 0)) / np.sum(np.abs(x)) if np.sum(np.abs(x)) > 0 else 0, raw=True
    )
    
    df['momentum_directional_bias'] = (df['bullish_momentum'] - df['bearish_momentum']) / (df['bullish_momentum'] + df['bearish_momentum'] + 1e-8)
    
    # Acceleration Pattern Recognition
    df['momentum_acceleration'] = (df['close'] - df['close'].shift(2)) - 2 * (df['close'].shift(1) - df['close'].shift(3))
    df['deceleration_detection'] = df['abs_price_change'] - df['abs_price_change'].shift(1)
    df['acceleration_direction_alignment'] = np.sign(df['price_change']) * (df['abs_price_change'] - df['abs_price_change'].shift(1))
    
    # Multi-Timeframe Momentum Consistency
    df['short_medium_alignment'] = (df['close'] - df['close'].shift(5)) / (df['close'] - df['close'].shift(1) + 1e-8)
    
    df['momentum_persistence_score'] = df['price_change'].rolling(window=10).apply(
        lambda x: np.sum(x > 0), raw=True
    )
    
    df['momentum_regime_stability'] = df['price_change'].rolling(window=10).apply(
        lambda x: np.std(x) / (np.mean(np.abs(x)) + 1e-8), raw=True
    )
    
    # Volume Acceleration Dynamics
    df['volume_change_ratio'] = df['volume'] / (df['volume'].shift(1) + 1e-8)
    df['volume_acceleration'] = df['volume_change_ratio'] - df['volume_change_ratio'].shift(1)
    
    df['volume_momentum_persistence'] = df['volume'].rolling(window=5).apply(
        lambda x: np.sum(x[1:] > x[:-1]), raw=True
    )
    
    df['volume_spike_quality'] = df['volume'] / df['volume'].rolling(window=4).mean().shift(1)
    
    # Price-Volume Acceleration Alignment
    df['acceleration_congruence'] = np.sign(df['price_change']) * (df['volume_change_ratio'] - 1)
    
    df['volume_confirmed_momentum'] = df['price_change'] * (df['volume'] / df['volume'].rolling(window=5).mean())
    
    df['acceleration_divergence_detection'] = (
        df['abs_price_change'] / df['abs_price_change'].rolling(window=4).mean().shift(1) - 
        df['volume'] / df['volume'].rolling(window=4).mean().shift(1)
    )
    
    # Volume Distribution Acceleration
    df['volume_concentration_shift'] = (
        df['volume'].rolling(window=5).max() - df['volume'].rolling(window=5).min()
    ) / (df['volume'].rolling(window=5).mean() + 1e-8)
    
    df['volume_volatility_acceleration'] = (
        df['volume'].rolling(window=5).std() / 
        (df['volume'].shift(5).rolling(window=5).std() + 1e-8)
    )
    
    # Momentum-Volume Asymmetry Patterns
    def calculate_up_move_volume_efficiency(window_data):
        prices = window_data['close'].values
        volumes = window_data['volume'].values
        up_volume = 0
        total_volume = np.sum(volumes)
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                up_volume += volumes[i]
        
        return up_volume / (total_volume + 1e-8)
    
    def calculate_down_move_volume_intensity(window_data):
        prices = window_data['close'].values
        volumes = window_data['volume'].values
        down_volume = 0
        total_price_change = 0
        
        for i in range(1, len(prices)):
            if prices[i] < prices[i-1]:
                down_volume += volumes[i]
                total_price_change += abs(prices[i] - prices[i-1])
        
        return down_volume / (total_price_change + 1e-8)
    
    # Calculate rolling asymmetry metrics
    up_move_efficiency = []
    down_move_intensity = []
    
    for i in range(len(df)):
        if i >= 4:
            window_data = df.iloc[i-4:i+1][['close', 'volume']]
            up_move_efficiency.append(calculate_up_move_volume_efficiency(window_data))
            down_move_intensity.append(calculate_down_move_volume_intensity(window_data))
        else:
            up_move_efficiency.append(np.nan)
            down_move_intensity.append(np.nan)
    
    df['up_move_volume_efficiency'] = up_move_efficiency
    df['down_move_volume_intensity'] = down_move_intensity
    df['asymmetry_ratio'] = (
        (df['up_move_volume_efficiency'] - df['down_move_volume_intensity']) / 
        (df['up_move_volume_efficiency'] + df['down_move_volume_intensity'] + 1e-8)
    )
    
    # Acceleration Asymmetry Analysis
    df['positive_acceleration'] = (df['price_change'] > df['price_change'].shift(1)).astype(int)
    df['negative_acceleration'] = (df['price_change'] < df['price_change'].shift(1)).astype(int)
    
    df['positive_acceleration_volume'] = df['volume'] * df['positive_acceleration']
    df['negative_acceleration_volume'] = df['volume'] * df['negative_acceleration']
    
    df['acceleration_volume_bias'] = (
        (df['positive_acceleration_volume'] - df['negative_acceleration_volume']) / 
        (df['positive_acceleration_volume'] + df['negative_acceleration_volume'] + 1e-8)
    )
    
    # Multi-Scale Asymmetry Patterns
    df['asymmetry_persistence'] = df['asymmetry_ratio'].rolling(window=10).apply(
        lambda x: np.sum(x > 0), raw=True
    )
    
    df['asymmetry_regime_stability'] = df['asymmetry_ratio'].rolling(window=10).apply(
        lambda x: np.std(x) / (np.mean(np.abs(x)) + 1e-8), raw=True
    )
    
    # Regime-Based Momentum Classification
    df['momentum_strength'] = df['abs_price_change'].rolling(window=10).mean()
    momentum_quantiles = df['momentum_strength'].quantile([0.4, 0.6])
    
    df['high_momentum_regime'] = (df['momentum_strength'] > momentum_quantiles[0.6]).astype(int)
    df['low_momentum_regime'] = (df['momentum_strength'] < momentum_quantiles[0.4]).astype(int)
    
    # Volume Acceleration Regimes
    df['accelerating_volume'] = (
        (df['volume_change_ratio'] > 1.2) & 
        (df['volume_change_ratio'].shift(1) > 1.1)
    ).astype(int)
    
    df['decelerating_volume'] = (
        (df['volume_change_ratio'] < 0.8) & 
        (df['volume_change_ratio'].shift(1) < 0.9)
    ).astype(int)
    
    df['stable_volume'] = (
        (df['volume'] / df['volume'].rolling(window=4).mean().shift(1)).between(0.9, 1.1)
    ).astype(int)
    
    # Combined factor calculation
    factors = [
        'momentum_directional_bias',
        'momentum_acceleration', 
        'acceleration_direction_alignment',
        'momentum_persistence_score',
        'volume_acceleration',
        'volume_confirmed_momentum',
        'asymmetry_ratio',
        'acceleration_volume_bias',
        'high_momentum_regime',
        'accelerating_volume'
    ]
    
    # Normalize and combine factors
    factor_values = df[factors].copy()
    
    # Remove any infinite values
    factor_values = factor_values.replace([np.inf, -np.inf], np.nan)
    
    # Z-score normalization for numerical factors
    numerical_factors = [f for f in factors if f not in ['high_momentum_regime', 'accelerating_volume']]
    for factor in numerical_factors:
        mean_val = factor_values[factor].mean()
        std_val = factor_values[factor].std()
        if std_val > 0:
            factor_values[factor] = (factor_values[factor] - mean_val) / std_val
    
    # Final factor combination with weights
    weights = {
        'momentum_directional_bias': 0.15,
        'momentum_acceleration': 0.12,
        'acceleration_direction_alignment': 0.10,
        'momentum_persistence_score': 0.08,
        'volume_acceleration': 0.10,
        'volume_confirmed_momentum': 0.15,
        'asymmetry_ratio': 0.12,
        'acceleration_volume_bias': 0.08,
        'high_momentum_regime': 0.05,
        'accelerating_volume': 0.05
    }
    
    # Calculate weighted factor score
    final_factor = pd.Series(0.0, index=df.index)
    for factor, weight in weights.items():
        final_factor += factor_values[factor] * weight
    
    return final_factor
