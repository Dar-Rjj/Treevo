import pandas as pd
import numpy as np
def heuristics_v2(df):
    import pandas as pd

    # Calculate Intraday Price Movement
    df['intraday_movement'] = df['high'] - df['low']
    df['prev_day_range'] = df['high'].shift(1) - df['low'].shift(1)
    df['movement_greater'] = df['intraday_movement'] > df['prev_day_range']

    # Analyze Volume Changes
    df['volume_change'] = df['volume'] > df['volume'].shift(1)

    # Combine Momentum and Volume Indicators
    conditions = [
        (df['intraday_movement'] > 0) & (df['movement_greater']) & (df['volume_change']),
        (df['intraday_movement'] > 0) & (df['movement_greater']) & (~df['volume_change']),
        (df['intraday_movement'] <= 0) | (~df['movement_greater']) & (df['volume_change']),
        (df['intraday_movement'] <= 0) | (~df['movement_greater']) & (~df['volume_change'])
    ]
    choices = [3, 2, -3, -2]  # 3: Strong Buy, 2: Weak Buy, -3: Strong Sell, -2: Weak Sell

    df['factor'] = pd.np.select(conditions, choices, default=0)

    return df['factor']
