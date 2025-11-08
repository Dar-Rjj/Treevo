import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Momentum-Volume Regime Factor
    Combines multi-timeframe momentum with volume-price alignment and volatility regime adaptation
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    epsilon = 1e-8
    
    # Multi-Timeframe Momentum Extraction
    # Intraday Momentum Component
    data['intraday_return'] = (data['close'] - data['open']) / (data['high'] - data['low'] + epsilon)
    data['intraday_direction'] = np.sign(data['close'] - data['open'])
    data['intraday_strength'] = np.abs(data['intraday_return'])
    
    # Short-term Momentum (3-day)
    data['close_3d_ago'] = data['close'].shift(3)
    data['high_3d_max'] = data['high'].rolling(window=4, min_periods=1).max()
    data['low_3d_min'] = data['low'].rolling(window=4, min_periods=1).min()
    data['short_momentum'] = (data['close'] - data['close_3d_ago']) / (data['high_3d_max'] - data['low_3d_min'] + epsilon)
    
    # Direction consistency for short-term
    data['daily_return'] = data['close'].pct_change()
    data['daily_direction'] = np.sign(data['daily_return'])
    data['direction_consistency_3d'] = data['daily_direction'].rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    )
    
    # Medium-term Momentum (10-day)
    data['close_10d_ago'] = data['close'].shift(10)
    data['high_10d_max'] = data['high'].rolling(window=11, min_periods=1).max()
    data['low_10d_min'] = data['low'].rolling(window=11, min_periods=1).min()
    data['medium_momentum'] = (data['close'] - data['close_10d_ago']) / (data['high_10d_max'] - data['low_10d_min'] + epsilon)
    
    # Trend persistence for medium-term
    def longest_streak(series):
        if len(series) < 2:
            return 0
        current_streak = 1
        max_streak = 1
        for i in range(1, len(series)):
            if series.iloc[i] == series.iloc[i-1] and not np.isnan(series.iloc[i]):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        return max_streak
    
    data['trend_persistence'] = data['daily_direction'].rolling(window=10, min_periods=1).apply(
        longest_streak, raw=False
    )
    
    # Volume-Price Alignment Framework
    # Volume Momentum Analysis
    data['volume_change_ratio'] = data['volume'] / (data['volume'].shift(1) + epsilon)
    data['volume_acceleration'] = (data['volume_change_ratio']) / (data['volume_change_ratio'].shift(1) + epsilon)
    
    # Volume persistence
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['volume_above_avg'] = (data['volume'] > data['volume_20d_avg']).astype(int)
    data['volume_persistence'] = data['volume_above_avg'].rolling(window=20, min_periods=1).apply(
        lambda x: x[::-1].cumprod().sum(), raw=False
    )
    
    # Volume-Price Confirmation
    data['directional_alignment'] = np.sign(data['volume_change_ratio']) * data['intraday_direction']
    data['strength_confirmation'] = np.abs(data['volume_change_ratio']) * data['intraday_strength']
    
    # Consistency score
    data['positive_alignment'] = (data['directional_alignment'] > 0).astype(int)
    data['consistency_score'] = data['positive_alignment'].rolling(window=5, min_periods=1).sum()
    
    # Adaptive Signal Weighting
    # Volatility-Regime Adjustment
    data['daily_range'] = data['high'] - data['low']
    data['recent_volatility'] = data['daily_range'].rolling(window=5, min_periods=1).mean()
    data['volatility_20d_avg'] = data['daily_range'].rolling(window=20, min_periods=1).mean()
    data['volatility_regime'] = data['daily_range'] / (data['volatility_20d_avg'] + epsilon)
    
    # Exponential Decay Application
    def decay_weighted_momentum(series, decay=0.95):
        weights = np.array([decay ** i for i in range(len(series))])
        return np.sum(weights * series) / np.sum(weights)
    
    data['decay_weighted_intraday'] = data['intraday_return'].rolling(window=5, min_periods=1).apply(
        lambda x: decay_weighted_momentum(x), raw=False
    )
    
    # Adaptive decay based on volatility regime
    data['adaptive_decay'] = 0.95 - 0.1 * (data['volatility_regime'] - 1).clip(-0.5, 0.5)
    
    # Final Factor Integration
    # Multi-timeframe Combination
    # Weights determined by volume confirmation strength
    volume_weight = data['strength_confirmation'].rolling(window=5, min_periods=1).mean()
    normalized_volume_weight = volume_weight / (volume_weight.rolling(window=20, min_periods=1).std() + epsilon)
    
    # Direction consistency across timeframes
    timeframe_alignment = (
        (data['intraday_direction'] == np.sign(data['short_momentum'])).astype(int) +
        (data['intraday_direction'] == np.sign(data['medium_momentum'])).astype(int) +
        (np.sign(data['short_momentum']) == np.sign(data['medium_momentum'])).astype(int)
    ) / 3.0
    
    # Weighted average of momentum components
    intraday_weight = 0.4 * normalized_volume_weight.clip(0, 2)
    short_weight = 0.35 * normalized_volume_weight.clip(0, 2)
    medium_weight = 0.25 * normalized_volume_weight.clip(0, 2)
    
    combined_momentum = (
        intraday_weight * data['intraday_return'] +
        short_weight * data['short_momentum'] +
        medium_weight * data['medium_momentum']
    )
    
    # Volume-Persistence Enhancement
    volume_persistence_enhancement = (1 + 0.1 * data['volume_persistence']) * (1 + 0.05 * data['volume_acceleration'])
    volume_direction_confirmation = np.sign(data['directional_alignment'])
    
    # Regime-Adaptive Output
    volatility_adjustment = 1 / (data['recent_volatility'] + epsilon)
    
    # Final factor calculation
    final_factor = (
        combined_momentum * 
        timeframe_alignment * 
        volume_persistence_enhancement * 
        volume_direction_confirmation * 
        volatility_adjustment
    )
    
    # Apply directional consistency filter
    consistency_filter = (data['direction_consistency_3d'] >= 2).astype(float)
    final_factor = final_factor * consistency_filter
    
    return final_factor
