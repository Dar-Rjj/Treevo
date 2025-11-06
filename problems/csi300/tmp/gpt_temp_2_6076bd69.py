import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Divergence
    # Short-term momentum (5-day return)
    short_momentum = data['close'].pct_change(periods=5)
    
    # Medium-term momentum (10-day return)
    medium_momentum = data['close'].pct_change(periods=10)
    
    # Momentum divergence ratio (5-day/10-day - 1)
    momentum_divergence = (short_momentum / medium_momentum) - 1
    momentum_divergence = momentum_divergence.replace([np.inf, -np.inf], np.nan)
    
    # Volume-Weighted Confirmation
    # Daily return
    daily_return = data['close'].pct_change()
    
    # Volume-weighted price changes (daily return × volume)
    volume_weighted_changes = daily_return * data['volume']
    
    # 5-day sum of volume-weighted changes
    volume_5day = volume_weighted_changes.rolling(window=5, min_periods=3).sum()
    
    # 10-day sum of volume-weighted changes
    volume_10day = volume_weighted_changes.rolling(window=10, min_periods=5).sum()
    
    # Volume acceleration ratio (5-day/10-day - 1)
    volume_acceleration = (volume_5day / volume_10day) - 1
    volume_acceleration = volume_acceleration.replace([np.inf, -np.inf], np.nan)
    
    # Intraday Strength Analysis
    # Strength ratio ((Close - Open)/(High - Low))
    strength_ratio = (data['close'] - data['open']) / (data['high'] - data['low'])
    strength_ratio = strength_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Strength persistence (max consecutive same-sign days)
    def max_consecutive_same_sign(series):
        signs = np.sign(series)
        max_streak = 0
        current_streak = 0
        current_sign = 0
        
        for sign in signs:
            if not np.isnan(sign):
                if sign == current_sign:
                    current_streak += 1
                else:
                    current_streak = 1
                    current_sign = sign
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
                current_sign = 0
        return max_streak
    
    # Calculate rolling strength persistence
    strength_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 20:  # Use 20-day window for calculation
            window_data = strength_ratio.iloc[i-20:i+1]
            strength_persistence.iloc[i] = max_consecutive_same_sign(window_data)
        else:
            strength_persistence.iloc[i] = np.nan
    
    # Daily price range (High - Low)
    daily_range = data['high'] - data['low']
    
    # Signal Integration
    # Primary signal: Momentum divergence × Volume acceleration
    primary_signal = momentum_divergence * volume_acceleration
    
    # Confirmation signal: Strength ratio × Strength persistence × Daily range
    confirmation_signal = strength_ratio * strength_persistence * daily_range
    
    # Final alpha: Primary signal × Confirmation signal
    final_alpha = primary_signal * confirmation_signal
    
    return final_alpha
