import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Weighted Multi-Timeframe Momentum with Volume Confirmation alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Calculation
    # Ultra-Short Term (1-2 days)
    data['price_momentum_1d'] = data['close'] - data['close'].shift(1)
    data['intraday_momentum'] = data['close'] - data['open']
    data['range_momentum'] = (data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1))
    
    # Short-Term (3-5 days)
    data['price_momentum_3d'] = data['close'] - data['close'].shift(3)
    
    # Calculate rolling average range for 3 days
    data['avg_range_3d'] = ((data['high'] - data['low']) + 
                           (data['high'].shift(1) - data['low'].shift(1)) + 
                           (data['high'].shift(2) - data['low'].shift(2))) / 3
    
    # Momentum consistency (count positive returns in last 3 days)
    data['positive_returns_3d'] = ((data['close'] > data['close'].shift(1)).astype(int) + 
                                  (data['close'].shift(1) > data['close'].shift(2)).astype(int) + 
                                  (data['close'].shift(2) > data['close'].shift(3)).astype(int))
    
    # Medium-Term (6-10 days)
    data['price_momentum_10d'] = data['close'] - data['close'].shift(9)
    
    # Trend strength
    rolling_high = data['high'].rolling(window=10, min_periods=10).max()
    rolling_low = data['low'].rolling(window=10, min_periods=10).min()
    data['trend_strength'] = data['price_momentum_10d'] / (rolling_high - rolling_low)
    
    # Direction persistence (longest streak of same return direction)
    returns = data['close'].pct_change()
    direction = np.sign(returns)
    streak = []
    current_streak = 1
    for i in range(len(direction)):
        if i == 0 or direction.iloc[i] != direction.iloc[i-1] or direction.iloc[i] == 0:
            current_streak = 1
        else:
            current_streak += 1
        streak.append(current_streak)
    data['direction_persistence'] = streak
    
    # Volatility Assessment
    # Range-Based Volatility
    data['short_term_vol'] = ((data['high'] - data['low']) + 
                             (data['high'].shift(1) - data['low'].shift(1)) + 
                             (data['high'].shift(2) - data['low'].shift(2)))
    
    data['medium_term_vol'] = (data['high'] - data['low']).rolling(window=10, min_periods=10).sum()
    data['volatility_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    
    # Volatility regimes
    data['vol_regime'] = np.select(
        [data['volatility_ratio'] > 1.2, 
         data['volatility_ratio'] < 0.8],
        ['high', 'low'],
        default='normal'
    )
    
    # Volatility-Weighted Momentum
    data['vw_momentum_3d'] = data['price_momentum_3d'] / data['short_term_vol']
    data['vw_momentum_10d'] = data['price_momentum_10d'] / data['medium_term_vol']
    data['vol_adjusted_trend'] = data['trend_strength'] * (1 / data['volatility_ratio'])
    
    # Volume Analysis
    data['volume_change'] = data['volume'] - data['volume'].shift(1)
    data['volume_direction'] = np.sign(data['volume_change'])
    
    # Volume streak (consecutive days with same volume direction)
    volume_streak = []
    current_vstreak = 1
    for i in range(len(data['volume_direction'])):
        if i == 0 or data['volume_direction'].iloc[i] != data['volume_direction'].iloc[i-1]:
            current_vstreak = 1
        else:
            current_vstreak += 1
        volume_streak.append(current_vstreak)
    data['volume_streak'] = volume_streak
    
    # Volume-Momentum Alignment
    data['alignment_score'] = np.sign(data['price_momentum_1d']) * np.sign(data['volume_change'])
    data['alignment_streak'] = data['alignment_score'].rolling(window=5, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] and x.iloc[i] > 0]), 
        raw=False
    )
    data['alignment_confidence'] = data['alignment_streak'] * abs(data['price_momentum_1d'])
    
    # Volume regimes
    data['volume_ratio_3d_10d'] = ((data['volume'] + data['volume'].shift(1) + data['volume'].shift(2)) / 
                                  data['volume'].rolling(window=10, min_periods=10).sum())
    data['volume_regime'] = np.select(
        [data['volume_ratio_3d_10d'] > 1.1, 
         data['volume_ratio_3d_10d'] < 0.9],
        ['high', 'low'],
        default='normal'
    )
    
    # Factor Construction
    # Core Momentum Signal
    data['combined_momentum'] = (0.4 * data['price_momentum_1d'] + 
                               0.35 * data['vw_momentum_3d'] + 
                               0.25 * data['vw_momentum_10d'])
    
    # Volume Confirmation
    data['base_factor'] = data['combined_momentum'] * (1 + 0.1 * data['alignment_score'])
    data['streak_enhanced'] = data['base_factor'] * (1 + 0.05 * np.minimum(data['volume_streak'], 5))
    
    # Volatility Adaptation
    regime_multipliers = {
        'high': 0.7,
        'normal': 1.0,
        'low': 1.3
    }
    data['vol_multiplier'] = data['vol_regime'].map(regime_multipliers)
    data['vol_adjusted_factor'] = data['streak_enhanced'] * data['vol_multiplier']
    data['vol_context_factor'] = data['vol_adjusted_factor'] * (2 - data['volatility_ratio'])
    
    # Volume Regime Integration
    volume_scaling = {
        'high': 1.2,
        'normal': 1.0,
        'low': 0.8
    }
    data['volume_multiplier'] = data['volume_regime'].map(volume_scaling)
    data['volume_scaled_factor'] = data['vol_context_factor'] * data['volume_multiplier']
    
    # Volume-Persistence Interaction
    data['volume_persistence_interaction'] = np.select(
        [(data['volume_regime'] == 'high') & (data['alignment_confidence'] > data['alignment_confidence'].quantile(0.7)), 
         (data['volume_regime'] == 'low') & (data['alignment_confidence'] < data['alignment_confidence'].quantile(0.3))],
        [1.1, 0.9],
        default=1.0
    )
    data['volume_persisted_factor'] = data['volume_scaled_factor'] * data['volume_persistence_interaction']
    
    # Momentum Acceleration
    data['acceleration_measure'] = data['vw_momentum_3d'] - data['vw_momentum_10d']
    data['acceleration_direction'] = np.sign(data['acceleration_measure'])
    data['momentum_confirmed_factor'] = data['volume_persisted_factor'] * (1 + 0.1 * data['acceleration_direction'])
    
    # Final Alpha Output
    alpha_factor = data['momentum_confirmed_factor']
    
    return alpha_factor
