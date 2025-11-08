import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Trend Momentum Efficiency
    # Trend Quality Assessment
    data['short_term_trend'] = data['close'] - data['close'].shift(3)
    data['medium_term_trend'] = data['close'] - data['close'].shift(8)
    data['trend_consistency'] = np.sign(data['short_term_trend']) * np.sign(data['medium_term_trend'])
    
    # Trend efficiency calculation
    data['net_movement'] = abs(data['close'] - data['close'].shift(5))
    data['total_oscillation'] = (
        abs(data['high'] - data['low']) + 
        abs(data['high'].shift(1) - data['low'].shift(1)) +
        abs(data['high'].shift(2) - data['low'].shift(2)) +
        abs(data['high'].shift(3) - data['low'].shift(3)) +
        abs(data['high'].shift(4) - data['low'].shift(4))
    )
    data['trend_efficiency'] = data['net_movement'] / (data['total_oscillation'] + 1e-8)
    
    # Volume-Price Synchronization
    # Volume trend dynamics
    data['volume_change'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_change_sign'] = np.sign(data['volume_change'])
    
    # Calculate volume persistence (3-day consecutive same sign)
    volume_persistence = []
    for i in range(len(data)):
        if i < 2:
            volume_persistence.append(1)
        else:
            current_sign = data['volume_change_sign'].iloc[i]
            if (current_sign == data['volume_change_sign'].iloc[i-1] == 
                data['volume_change_sign'].iloc[i-2]):
                volume_persistence.append(3)
            elif current_sign == data['volume_change_sign'].iloc[i-1]:
                volume_persistence.append(2)
            else:
                volume_persistence.append(1)
    data['volume_persistence'] = volume_persistence
    data['volume_trend_strength'] = abs(data['volume_change']) * data['volume_persistence']
    
    # Price-volume alignment
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['directional_alignment'] = np.sign(data['price_change']) * np.sign(data['volume_change'])
    data['magnitude_sync'] = abs(data['price_change']) * abs(data['volume_change'])
    data['sync_score'] = data['directional_alignment'] * data['magnitude_sync']
    
    # Momentum Quality Enhancement
    # Momentum persistence analysis
    data['returns'] = data['close'].pct_change()
    data['return_sign'] = np.sign(data['returns'])
    
    # Calculate momentum direction streak (5-day)
    momentum_streak = []
    for i in range(len(data)):
        if i < 4:
            momentum_streak.append(1)
        else:
            current_sign = data['return_sign'].iloc[i]
            streak = 1
            for j in range(1, 5):
                if data['return_sign'].iloc[i-j] == current_sign:
                    streak += 1
                else:
                    break
            momentum_streak.append(streak)
    data['momentum_streak'] = momentum_streak
    
    # Momentum magnitude consistency (std of returns over streak period)
    momentum_consistency = []
    for i in range(len(data)):
        if i < 4:
            momentum_consistency.append(1.0)
        else:
            streak_len = min(data['momentum_streak'].iloc[i], 5)
            period_returns = data['returns'].iloc[i-streak_len+1:i+1]
            if len(period_returns) > 1:
                consistency = streak_len / (np.std(period_returns) + 1e-8)
            else:
                consistency = 1.0
            momentum_consistency.append(consistency)
    data['momentum_quality'] = momentum_consistency
    
    # Acceleration quality
    data['momentum_change'] = data['returns'] - data['returns'].shift(1)
    data['acceleration_sign'] = np.sign(data['momentum_change'])
    
    # Calculate acceleration persistence (3-day)
    accel_persistence = []
    for i in range(len(data)):
        if i < 2:
            accel_persistence.append(1)
        else:
            current_sign = data['acceleration_sign'].iloc[i]
            if (current_sign == data['acceleration_sign'].iloc[i-1] == 
                data['acceleration_sign'].iloc[i-2]):
                accel_persistence.append(3)
            elif current_sign == data['acceleration_sign'].iloc[i-1]:
                accel_persistence.append(2)
            else:
                accel_persistence.append(1)
    data['accel_persistence'] = accel_persistence
    data['quality_acceleration'] = data['momentum_change'] * data['accel_persistence']
    
    # Regime Transition Detection
    data['local_volatility'] = data['high'] - data['low']
    data['volatility_change'] = data['local_volatility'] / (data['local_volatility'].shift(2) + 1e-8) - 1
    data['regime_shift'] = abs(data['volatility_change']) > 0.5
    
    # Volume regime transitions
    data['volume_spike'] = data['volume'] / (data['volume'].shift(1) + 1e-8) > 2.0
    data['volume_collapse'] = data['volume'] / (data['volume'].shift(1) + 1e-8) < 0.5
    data['transition_score'] = (data['volume_spike'].astype(int) - data['volume_collapse'].astype(int)) * data['price_change']
    
    # Price Level Context
    data['daily_range'] = data['high'] - data['low']
    data['range_position'] = (data['close'] - data['low']) / (data['daily_range'] + 1e-8)
    
    # Multi-day high/low proximity
    data['high_10d'] = data['high'].rolling(window=10, min_periods=1).max()
    data['low_10d'] = data['low'].rolling(window=10, min_periods=1).min()
    data['dist_to_high'] = (data['high_10d'] - data['close']) / data['high_10d']
    data['dist_to_low'] = (data['close'] - data['low_10d']) / data['low_10d']
    data['high_low_proximity'] = np.minimum(data['dist_to_high'], data['dist_to_low'])
    data['position_strength'] = data['range_position'] * data['high_low_proximity']
    
    # Breakout context
    data['near_high'] = data['close'] > 0.9 * data['high_10d']
    data['near_low'] = data['close'] < 1.1 * data['low_10d']
    
    # Efficiency-Weighted Synthesis
    # Component quality weighting
    trend_weight = data['trend_efficiency']
    sync_weight = data['sync_score']
    momentum_weight = data['momentum_quality']
    
    # Core efficiency momentum
    quality_trend = (data['short_term_trend'] + data['medium_term_trend']) * trend_weight
    synchronized_momentum = data['quality_acceleration'] * sync_weight
    efficiency_momentum = quality_trend * synchronized_momentum
    
    # Regime context application
    # Normal regime: weighted average
    normal_factor = (efficiency_momentum * 0.4 + 
                    data['sync_score'] * 0.3 + 
                    data['momentum_quality'] * 0.3)
    
    # Transition regime: boost synchronization
    transition_factor = (efficiency_momentum * 0.2 + 
                        data['sync_score'] * 0.6 + 
                        data['momentum_quality'] * 0.2)
    
    # High volatility: emphasize momentum quality
    high_vol_factor = (efficiency_momentum * 0.3 + 
                      data['sync_score'] * 0.2 + 
                      data['momentum_quality'] * 0.5)
    
    # Apply regime selection
    regime_factor = normal_factor.copy()
    regime_factor[data['regime_shift']] = transition_factor[data['regime_shift']]
    regime_factor[data['local_volatility'] > data['local_volatility'].rolling(window=10).mean()] = high_vol_factor
    
    # Incorporate price level context
    context_adjusted = regime_factor * data['position_strength']
    
    # Signal refinement
    # Persistence filter (2-day consistent direction)
    factor_sign = np.sign(context_adjusted)
    persistence_filter = (factor_sign == factor_sign.shift(1))
    
    # Magnitude threshold
    magnitude_threshold = abs(context_adjusted) > abs(context_adjusted).rolling(window=20).mean()
    
    # Final alpha
    final_alpha = context_adjusted.copy()
    final_alpha[~(persistence_filter & magnitude_threshold)] = 0
    
    return final_alpha
