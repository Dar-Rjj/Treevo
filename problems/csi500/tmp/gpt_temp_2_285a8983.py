import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Scaled Volume-Momentum Persistence with Adaptive Decay alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Core Price Components
    data['intraday_momentum'] = data['close'] - data['open']
    data['daily_range'] = data['high'] - data['low']
    data['price_direction'] = np.sign(data['intraday_momentum'])
    
    # Multi-Timeframe Analysis
    # Very Short-Term (1-3 days)
    data['momentum_3d'] = data['intraday_momentum'].rolling(window=3, min_periods=3).sum()
    data['range_3d'] = data['daily_range'].rolling(window=3, min_periods=3).mean()
    data['volume_3d'] = data['volume'].rolling(window=3, min_periods=3).mean()
    
    # Short-Term (5-10 days)
    data['momentum_5d'] = data['intraday_momentum'].rolling(window=5, min_periods=5).sum()
    data['range_5d'] = data['daily_range'].rolling(window=5, min_periods=5).mean()
    data['volume_5d'] = data['volume'].rolling(window=5, min_periods=5).mean()
    
    # Medium-Term (15-20 days)
    data['momentum_20d'] = data['intraday_momentum'].rolling(window=20, min_periods=20).sum()
    data['range_20d'] = data['daily_range'].rolling(window=20, min_periods=20).mean()
    data['volume_20d'] = data['volume'].rolling(window=20, min_periods=20).mean()
    
    # Volatility Scaling System
    data['volatility_ratio'] = data['range_5d'] / data['range_20d']
    data['volatility_regime'] = np.select(
        [data['volatility_ratio'] > 1.2, data['volatility_ratio'] < 0.8],
        ['high', 'low'],
        default='normal'
    )
    
    data['vol_scaled_momentum'] = data['momentum_5d'] / data['range_5d']
    data['vol_scaled_volume'] = data['volume'] / data['daily_range']
    data['range_efficiency'] = np.abs(data['intraday_momentum']) / data['daily_range']
    
    # Volume-Momentum Persistence
    # Direction Persistence
    data['direction_change'] = data['price_direction'] != data['price_direction'].shift(1)
    data['persistence_days'] = data.groupby(data['direction_change'].cumsum()).cumcount() + 1
    data['persistence_strength'] = data['persistence_days'] * np.abs(data['intraday_momentum'])
    
    # Volume Confirmation
    data['volume_direction_alignment'] = data['price_direction'] * np.sign(data['volume'] - data['volume_5d'])
    data['alignment_change'] = data['volume_direction_alignment'] <= 0
    data['alignment_streak'] = data.groupby(data['alignment_change'].cumsum()).cumcount() + 1
    data['confirmation_strength'] = data['alignment_streak'] * np.abs(data['volume'] - data['volume_5d'])
    
    # Volume-Volatility Interaction
    data['volume_density'] = data['volume'] / data['daily_range']
    data['volume_density_5d'] = data['volume_density'].rolling(window=5, min_periods=5).mean()
    data['volume_density_trend'] = data['volume_density'] / data['volume_density_5d']
    data['efficiency_volume_alignment'] = data['range_efficiency'] * data['volume_density']
    
    # Adaptive Exponential Decay
    def exponential_weighted_sum(series, window=5, decay=0.9):
        weights = np.array([decay ** i for i in range(window)])[::-1]
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.sum(x * weights[:len(x)]), raw=True
        )
    
    data['weighted_momentum'] = exponential_weighted_sum(data['intraday_momentum'])
    data['weighted_volume'] = exponential_weighted_sum(data['volume'])
    
    # Acceleration Detection
    data['momentum_acceleration'] = data['momentum_5d'] - data['momentum_20d']
    data['acceleration_direction'] = np.sign(data['momentum_acceleration'])
    data['acceleration_strength'] = np.abs(data['momentum_acceleration']) / data['range_20d']
    
    # Decay-Adjusted Persistence
    data['weighted_persistence'] = exponential_weighted_sum(data['persistence_strength'])
    data['weighted_alignment'] = exponential_weighted_sum(data['confirmation_strength'])
    
    # Range-Based Normalization
    data['directional_efficiency'] = np.where(
        data['close'] > data['open'],
        (data['close'] - data['open']) / data['daily_range'],
        (data['open'] - data['close']) / data['daily_range']
    )
    
    # Efficiency persistence
    data['efficiency_high'] = data['directional_efficiency'] > 0.5
    data['efficiency_change'] = ~data['efficiency_high']
    data['efficiency_persistence'] = data.groupby(data['efficiency_change'].cumsum()).cumcount() + 1
    
    data['normalized_momentum'] = data['momentum_5d'] / data['range_20d']
    data['normalized_volume'] = data['volume_5d'] / data['range_20d']
    data['normalized_persistence'] = data['persistence_strength'] / data['range_5d']
    
    # Relative Strength Positioning
    data['range_relative'] = data['daily_range'] / data['range_20d']
    data['volume_relative'] = data['volume'] / data['volume_20d']
    data['strength_ratio'] = np.abs(data['momentum_5d']) / data['range_5d']
    
    # Factor Construction
    # Base Momentum Component
    data['base_momentum'] = data['vol_scaled_momentum'] * data['range_efficiency']
    data['persistence_enhanced'] = data['base_momentum'] * (1 + data['weighted_persistence'] / 10)
    data['volume_confirmed'] = data['persistence_enhanced'] * (1 + data['weighted_alignment'] / 8)
    
    # Regime-Adaptive Scaling
    # Volatility Regime Multipliers
    volatility_multiplier = np.select(
        [data['volatility_regime'] == 'high', data['volatility_regime'] == 'low'],
        [0.6, 1.4],
        default=1.0
    )
    
    # Volume Regime Adjustments
    volume_density_quantile = data['volume_density'].rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    volume_multiplier = np.select(
        [volume_density_quantile > 0.7, volume_density_quantile < 0.3],
        [1.3, 0.7],
        default=1.0
    )
    
    # Efficiency-Based Boosts
    efficiency_multiplier = np.select(
        [data['range_efficiency'] > 0.7, data['range_efficiency'] < 0.3],
        [1.2, 0.8],
        default=1.0
    )
    
    # Acceleration Finalizer
    acceleration_multiplier = 1 + (0.15 * data['acceleration_direction'])
    strength_amplifier = 1 + (0.1 * data['acceleration_strength'])
    
    # Final Factor Calculation
    data['alpha_factor'] = (
        data['volume_confirmed'] * 
        volatility_multiplier * 
        volume_multiplier * 
        efficiency_multiplier * 
        acceleration_multiplier * 
        strength_amplifier
    )
    
    return data['alpha_factor']
