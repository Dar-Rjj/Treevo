import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum with Volume Confirmation alpha factor
    
    Parameters:
    data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    
    Returns:
    Series: alpha factor values indexed by date
    """
    
    # Price Components
    # Intraday Momentum
    intraday_momentum = (data['close'] - data['open']) / (data['high'] - data['low'])
    intraday_momentum = intraday_momentum.replace([np.inf, -np.inf], np.nan)
    
    # Short-term Momentum
    close_lag2 = data['close'].shift(2)
    high_low_sum = (data['high'] - data['low']) + (data['high'].shift(1) - data['low'].shift(1))
    short_term_momentum = (data['close'] - close_lag2) / high_low_sum
    short_term_momentum = short_term_momentum.replace([np.inf, -np.inf], np.nan)
    
    # Combined Momentum
    combined_momentum = (2 * intraday_momentum + short_term_momentum) / 3
    
    # Volume Components
    # Volume Direction
    volume_direction = np.sign(data['volume'] - data['volume'].shift(1))
    
    # Price-Volume Alignment
    price_change_direction = np.sign(data['close'] - data['close'].shift(1))
    price_volume_alignment = price_change_direction * volume_direction
    
    # Volume Streak
    volume_streak = pd.Series(index=data.index, dtype=float)
    current_streak = 0
    prev_direction = 0
    
    for i in range(len(data)):
        if i == 0:
            volume_streak.iloc[i] = 0
            prev_direction = volume_direction.iloc[i]
            continue
            
        current_direction = volume_direction.iloc[i]
        if current_direction == prev_direction and current_direction != 0:
            current_streak += 1
        else:
            current_streak = 1 if current_direction != 0 else 0
        
        volume_streak.iloc[i] = current_streak
        prev_direction = current_direction
    
    # Volatility Regime
    # Short-term Volatility (3-day)
    short_term_vol = (data['high'] - data['low'] + 
                     data['high'].shift(1) - data['low'].shift(1) + 
                     data['high'].shift(2) - data['low'].shift(2)) / 3
    
    # Medium-term Volatility (10-day)
    medium_term_vol = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 9:
            window = data.iloc[i-9:i+1]
            medium_term_vol.iloc[i] = (window['high'] - window['low']).sum() / 10
        else:
            medium_term_vol.iloc[i] = np.nan
    
    # Volatility Ratio
    volatility_ratio = short_term_vol / medium_term_vol
    
    # Regime Multiplier
    regime_multiplier = pd.Series(index=data.index, dtype=float)
    regime_multiplier[volatility_ratio > 1.1] = 0.7  # High Volatility
    regime_multiplier[(volatility_ratio >= 0.9) & (volatility_ratio <= 1.1)] = 1.0  # Normal Volatility
    regime_multiplier[volatility_ratio < 0.9] = 1.3  # Low Volatility
    
    # Volume Confirmation
    # Base Alignment
    base_alignment = price_volume_alignment
    
    # Streak Bonus
    streak_bonus = np.minimum(volume_streak * 0.1, 0.3)
    
    # Total Multiplier
    total_multiplier = 1 + base_alignment + streak_bonus
    
    # Final Alpha
    # Regime-Adjusted Momentum
    regime_adjusted_momentum = combined_momentum * regime_multiplier
    
    # Volume-Confirmed Alpha
    alpha = regime_adjusted_momentum * total_multiplier
    
    return alpha
