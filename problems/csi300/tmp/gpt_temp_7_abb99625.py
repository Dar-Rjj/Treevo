import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Momentum Acceleration with Volume Confirmation and Regime Switching alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Momentum Acceleration Calculation
    # Short-term momentum
    data['mom_3d'] = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    data['mom_6d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Momentum acceleration
    data['acceleration'] = data['mom_3d'] - data['mom_6d']
    
    # Acceleration persistence
    acc_direction = np.sign(data['acceleration'])
    persistence = []
    current_streak = 0
    for i in range(len(acc_direction)):
        if i == 0 or acc_direction.iloc[i] != acc_direction.iloc[i-1]:
            current_streak = 1
        else:
            current_streak += 1
        persistence.append(current_streak)
    data['acc_persistence'] = persistence
    
    # 2. Volume Confirmation Analysis
    # Volume trend assessment
    def volume_slope(window):
        if len(window) < 2:
            return 0
        x = np.arange(len(window))
        slope, _, _, _, _ = stats.linregress(x, window)
        return slope
    
    data['volume_slope_5d'] = data['volume'].rolling(window=5, min_periods=3).apply(
        volume_slope, raw=True
    )
    
    # Volume acceleration
    data['volume_acceleration'] = data['volume_slope_5d'] - data['volume_slope_5d'].shift(3)
    
    # Price-volume divergence analysis
    data['price_direction'] = np.sign(data['mom_3d'])
    data['volume_direction'] = np.sign(data['volume_slope_5d'])
    
    # Volume quality score
    conditions = [
        (data['price_direction'] == data['volume_direction']) & (abs(data['volume_slope_5d']) > data['volume_slope_5d'].rolling(20).std()),
        (data['price_direction'] == data['volume_direction']) & (abs(data['volume_slope_5d']) <= data['volume_slope_5d'].rolling(20).std()),
        (data['price_direction'] != data['volume_direction'])
    ]
    choices = [3, 2, 1]  # High, Medium, Low quality
    data['volume_quality'] = np.select(conditions, choices, default=2)
    
    # 3. Market Regime Detection
    # Price range compression
    data['high_10d'] = data['high'].rolling(window=10, min_periods=8).max()
    data['low_10d'] = data['low'].rolling(window=10, min_periods=8).min()
    data['range_compression'] = (data['high_10d'] - data['low_10d']) / data['close']
    
    # Trend strength measurement
    def price_slope(window):
        if len(window) < 2:
            return 0
        x = np.arange(len(window))
        slope, _, _, _, _ = stats.linregress(x, window)
        return abs(slope)
    
    data['trend_strength'] = data['close'].rolling(window=10, min_periods=8).apply(
        price_slope, raw=True
    ) / data['close'].rolling(window=10).mean()
    
    # Regime classification
    range_median = data['range_compression'].rolling(window=50).median()
    trend_median = data['trend_strength'].rolling(window=50).median()
    
    conditions_regime = [
        (data['trend_strength'] > trend_median) & (data['range_compression'] <= range_median * 1.2),
        (data['trend_strength'] <= trend_median * 0.8) & (data['range_compression'] <= range_median * 0.8),
        (data['trend_strength'] > trend_median) & (data['range_compression'] > range_median * 1.2),
        (data['trend_strength'] <= trend_median * 0.8) & (data['range_compression'] > range_median * 1.2)
    ]
    regime_labels = ['trending', 'consolidating', 'breakout', 'choppy']
    data['regime'] = np.select(conditions_regime, regime_labels, default='trending')
    
    # Regime persistence
    regime_persistence = []
    current_regime_streak = 0
    for i in range(len(data)):
        if i == 0 or data['regime'].iloc[i] != data['regime'].iloc[i-1]:
            current_regime_streak = 1
        else:
            current_regime_streak += 1
        regime_persistence.append(current_regime_streak)
    data['regime_persistence'] = regime_persistence
    
    # 4. Adaptive Factor Integration
    # Regime-adaptive momentum weighting
    regime_weights = {
        'trending': 1.2,
        'consolidating': 0.8,
        'breakout': 1.5,
        'choppy': 0.5
    }
    data['regime_weight'] = data['regime'].map(regime_weights)
    
    # Volume-confirmed acceleration signal
    volume_multipliers = {
        3: 1.0,   # High quality: full factor
        2: 0.6,   # Medium quality: 60% factor
        1: 0.2    # Low quality: 20% factor
    }
    data['volume_multiplier'] = data['volume_quality'].map(volume_multipliers)
    
    # Apply volume direction adjustment for divergent cases
    divergent_mask = data['price_direction'] != data['volume_direction']
    data['volume_multiplier'] = np.where(divergent_mask, -data['volume_multiplier'], data['volume_multiplier'])
    
    # Final alpha generation
    # Base acceleration signal adjusted by persistence
    base_signal = data['acceleration'] * (1 + 0.1 * np.log1p(data['acc_persistence']))
    
    # Apply regime weighting and volume confirmation
    final_alpha = base_signal * data['regime_weight'] * data['volume_multiplier']
    
    # Normalize the final alpha
    final_alpha = (final_alpha - final_alpha.rolling(window=50, min_periods=30).mean()) / final_alpha.rolling(window=50, min_periods=30).std()
    
    return final_alpha
