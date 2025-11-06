import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate daily range
    df['range'] = df['high'] - df['low']
    
    # Range Momentum Calculation
    # 5-day Range Return
    df['range_5d_return'] = (df['range'] / df['range'].shift(5)) - 1
    
    # 10-day Range Return
    df['range_10d_return'] = (df['range'] / df['range'].shift(10)) - 1
    
    # Range Direction Consistency
    df['range_change'] = df['range'] - df['range'].shift(1)
    df['range_direction'] = np.sign(df['range_change'])
    
    # Count consecutive same-direction range changes (3-day window)
    range_consistency = []
    for i in range(len(df)):
        if i < 2:
            range_consistency.append(0)
        else:
            window = df['range_direction'].iloc[max(0, i-2):i+1]
            if len(window) == 3 and len(set(window)) == 1:
                range_consistency.append(3)
            elif len(window) >= 2 and len(set(window.iloc[-2:])) == 1:
                range_consistency.append(2)
            else:
                range_consistency.append(1)
    df['range_direction_consistency'] = range_consistency
    
    # Volume Divergence Patterns
    # Volume Ratio: Mean(Volume[t-4:t]) / Mean(Volume[t-9:t])
    df['volume_ratio'] = (df['volume'].rolling(window=5, min_periods=1).mean() / 
                         df['volume'].rolling(window=10, min_periods=1).mean())
    
    # Volume Trend Slope: Linear regression slope of Volume[t-4:t]
    volume_slopes = []
    for i in range(len(df)):
        if i < 4:
            volume_slopes.append(0)
        else:
            window_volume = df['volume'].iloc[i-4:i+1].values
            if len(window_volume) == 5:
                try:
                    slope = linregress(range(5), window_volume)[0]
                    volume_slopes.append(slope)
                except:
                    volume_slopes.append(0)
            else:
                volume_slopes.append(0)
    df['volume_trend_slope'] = volume_slopes
    
    # Signal Combination - Momentum-Volume Products
    df['momentum_volume_5d'] = df['range_5d_return'] * df['volume_ratio']
    df['momentum_volume_10d'] = df['range_10d_return'] * df['volume_ratio']
    df['consistency_slope'] = df['range_direction_consistency'] * df['volume_trend_slope']
    
    # Final Alpha: Weighted composite of momentum-volume products
    # Weights: 0.4 for 5-day, 0.3 for 10-day, 0.3 for consistency-slope
    alpha = (0.4 * df['momentum_volume_5d'] + 
             0.3 * df['momentum_volume_10d'] + 
             0.3 * df['consistency_slope'])
    
    return alpha
