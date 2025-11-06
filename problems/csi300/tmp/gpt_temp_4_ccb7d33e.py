import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate momentum components
    df = df.copy()
    
    # Short-term momentum calculations
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # Momentum acceleration factor
    df['momentum_accel'] = (df['momentum_5'] - df['momentum_5'].shift(5)) / 5
    
    # Reversal conditions
    df['momentum_diff'] = df['momentum_5'] - df['momentum_10']
    df['reversal_signal'] = np.where(
        (df['momentum_accel'] > df['momentum_accel'].rolling(20).quantile(0.8)) & 
        (df['momentum_diff'] > 0), 
        -1,  # High positive acceleration with positive momentum diff → potential reversal
        np.where(
            (df['momentum_accel'] < df['momentum_accel'].rolling(20).quantile(0.2)) & 
            (df['momentum_diff'] < 0), 
            1,   # High negative acceleration with negative momentum diff → potential bounce
            0
        )
    )
    
    # Volume trend and surge
    df['volume_slope'] = df['volume'].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Volume confirmation
    df['volume_confirmation'] = np.where(
        (df['volume_ratio'] > 1.2) & (df['volume_slope'] > 0),
        1.5,  # Strong volume surge
        np.where(
            (df['volume_ratio'] < 0.8) & (df['volume_slope'] < 0),
            0.5,  # Weak volume
            1.0   # Normal volume
        )
    )
    
    # Range efficiency validation
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    df['daily_range'] = df['high'] - df['low']
    df['range_efficiency'] = df['daily_range'] / df['true_range']
    
    # Range efficiency validation score
    df['range_validation'] = np.where(
        df['range_efficiency'] > df['range_efficiency'].rolling(20).quantile(0.7),
        1.2,  # High efficiency
        np.where(
            df['range_efficiency'] < df['range_efficiency'].rolling(20).quantile(0.3),
            0.8,  # Low efficiency
            1.0   # Normal efficiency
        )
    )
    
    # Combine all components
    df['factor'] = (
        df['reversal_signal'] * 
        abs(df['momentum_accel']) * 
        df['volume_confirmation'] * 
        df['range_validation']
    )
    
    # Clean up and return
    result = df['factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
