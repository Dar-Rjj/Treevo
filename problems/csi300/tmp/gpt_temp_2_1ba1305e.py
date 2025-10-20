import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence Momentum factor combining multiple technical signals.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Price-Volume Divergence Momentum
    # Price momentum calculations
    data['price_momentum_5'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_20'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume momentum calculations
    data['volume_momentum_5'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_20'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Divergence signal - compare directions
    price_dir_5 = np.sign(data['price_momentum_5'])
    volume_dir_5 = np.sign(data['volume_momentum_5'])
    price_dir_20 = np.sign(data['price_momentum_20'])
    volume_dir_20 = np.sign(data['volume_momentum_20'])
    
    # Strength score based on divergence
    divergence_strength = (
        (price_dir_5 != volume_dir_5).astype(int) * 0.6 + 
        (price_dir_20 != volume_dir_20).astype(int) * 0.4
    ) * (abs(data['price_momentum_5']) + abs(data['price_momentum_20'])) / 2
    
    # 2. Volatility-Adjusted Return Reversal
    # Recent return
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    
    # Historical volatility (20-day)
    data['returns_daily'] = data['close'].pct_change()
    data['volatility_20d'] = data['returns_daily'].rolling(window=20).std()
    
    # Volatility regime adjustment
    vol_median = data['volatility_20d'].rolling(window=60).median()
    high_vol_regime = (data['volatility_20d'] > vol_median).astype(int)
    
    # Scale return by volatility - amplify in high volatility
    volatility_adjusted_return = data['return_3d'] * (
        1 + high_vol_regime * data['volatility_20d'] / vol_median
    )
    
    # 3. Intraday Strength Persistence
    # Daily trading range strength
    data['normalized_range'] = (data['high'] - data['low']) / data['close']
    data['distance_from_high'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    data['distance_from_low'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Range strength score
    range_strength = (1 - data['distance_from_high']) * (1 - data['distance_from_low'])
    
    # Multi-day pattern analysis
    strong_day = range_strength > range_strength.rolling(window=10).mean()
    persistence_streak = strong_day.rolling(window=5).sum()
    
    # Momentum continuity score
    momentum_continuity = persistence_streak * range_strength
    
    # 4. Volume-Weighted Price Acceleration
    # Price acceleration calculations
    data['price_return'] = data['close'].pct_change()
    data['price_acceleration'] = data['price_return'].diff()
    
    # Volume trend
    data['volume_trend'] = data['volume'] / data['volume'].rolling(window=10).mean()
    
    # Weighted acceleration factor
    weighted_acceleration = data['price_acceleration'] * data['volume_trend']
    
    # 5. Amplitude-Duration Momentum
    # Price movement characteristics
    data['daily_amplitude'] = (data['high'] - data['low']) / data['close']
    
    # Count consecutive up/down days
    up_days = (data['close'] > data['close'].shift(1)).astype(int)
    down_days = (data['close'] < data['close'].shift(1)).astype(int)
    
    # Rolling count of consecutive movements
    def consecutive_count(series):
        count = pd.Series(index=series.index, dtype=float)
        current_streak = 0
        for i in range(len(series)):
            if series.iloc[i]:
                current_streak += 1
            else:
                current_streak = 0
            count.iloc[i] = current_streak
        return count
    
    up_streak = consecutive_count(up_days)
    down_streak = consecutive_count(down_days)
    
    # Movement duration score
    movement_duration = up_streak - down_streak
    
    # Composite momentum score
    amplitude_duration_score = data['daily_amplitude'] * movement_duration
    
    # Final factor combination
    factor = (
        0.25 * divergence_strength +
        0.20 * volatility_adjusted_return +
        0.20 * momentum_continuity +
        0.20 * weighted_acceleration +
        0.15 * amplitude_duration_score
    )
    
    # Clean and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return factor
