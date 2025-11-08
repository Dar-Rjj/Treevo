import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Momentum Calculation
    # Normalized intraday return
    data['intraday_return'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_return'] = data['intraday_return'].replace([np.inf, -np.inf], np.nan)
    
    # Momentum direction (1 for up, -1 for down, 0 for flat)
    data['direction'] = np.sign(data['intraday_return'])
    data['direction'] = data['direction'].replace(0, np.nan)
    
    # Calculate consecutive days with same direction
    streak = []
    current_streak = 1
    current_direction = None
    
    for i in range(len(data)):
        if pd.isna(data['direction'].iloc[i]):
            streak.append(0)
            current_streak = 1
            current_direction = None
        elif current_direction is None:
            streak.append(1)
            current_direction = data['direction'].iloc[i]
        elif data['direction'].iloc[i] == current_direction:
            current_streak += 1
            streak.append(current_streak)
        else:
            current_streak = 1
            current_direction = data['direction'].iloc[i]
            streak.append(1)
    
    data['streak_length'] = streak
    
    # Calculate average normalized return magnitude during streak
    data['abs_intraday_return'] = data['intraday_return'].abs()
    
    # Rolling average of absolute returns for current streak period
    def rolling_streak_avg(series, streak_series):
        result = []
        for i in range(len(series)):
            if streak_series.iloc[i] > 0:
                start_idx = max(0, i - streak_series.iloc[i] + 1)
                avg_val = series.iloc[start_idx:i+1].mean()
                result.append(avg_val)
            else:
                result.append(0)
        return result
    
    data['streak_avg_return'] = rolling_streak_avg(data['abs_intraday_return'], data['streak_length'])
    
    # Momentum strength
    data['momentum_strength'] = data['streak_length'] * data['streak_avg_return']
    
    # Volume Confirmation Analysis
    # Volume efficiency
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low'])
    data['volume_efficiency'] = data['volume_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # 5-day average volume efficiency
    data['avg_volume_efficiency_5d'] = data['volume_efficiency'].rolling(window=5, min_periods=1).mean()
    
    # Volume confirmation
    data['volume_confirmation'] = data['volume_efficiency'] / data['avg_volume_efficiency_5d']
    data['volume_confirmation'] = data['volume_confirmation'].replace([np.inf, -np.inf], np.nan)
    
    # Volume breakout detection
    data['avg_volume_efficiency_10d'] = data['volume_efficiency'].rolling(window=10, min_periods=1).mean()
    data['volume_breakout'] = data['volume_efficiency'] > (2 * data['avg_volume_efficiency_10d'])
    
    # Signal Combination
    # Base composite
    data['base_composite'] = data['momentum_strength'] * data['volume_confirmation']
    
    # Enhanced composite
    data['enhanced_composite'] = data['base_composite']
    data.loc[data['volume_breakout'], 'enhanced_composite'] = 2 * data['base_composite']
    
    # Price Context Integration
    # 20-day high-low range
    data['high_20d'] = data['high'].rolling(window=20, min_periods=1).max()
    data['low_20d'] = data['low'].rolling(window=20, min_periods=1).min()
    
    # Position within 20-day range
    data['price_position'] = (data['close'] - data['low_20d']) / (data['high_20d'] - data['low_20d'])
    data['price_position'] = data['price_position'].replace([np.inf, -np.inf], np.nan)
    
    # Final adjustment
    data['final_composite'] = data['enhanced_composite'] * (1 + data['price_position'])
    
    # Signal Filtering
    # Apply only when ≥3 consecutive days same intraday direction
    data['filtered_signal'] = data['final_composite']
    data.loc[data['streak_length'] < 3, 'filtered_signal'] = 0
    
    return data['filtered_signal']
