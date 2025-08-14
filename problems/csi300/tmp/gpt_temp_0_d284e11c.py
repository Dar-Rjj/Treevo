import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()

    # Compute Rolling Sum of Returns
    df['5_day_sum'] = df['daily_return'].rolling(window=5).sum()
    df['20_day_sum'] = df['daily_return'].rolling(window=20).sum()
    df['rolling_sum_diff'] = df['20_day_sum'] - df['5_day_sum']

    # Determine Directional Movement
    df['PDM'] = 0
    df['NDM'] = 0
    for i in range(1, len(df)):
        if df['high'].iloc[i] > df['high'].iloc[i-1]:
            df['PDM'].iloc[i] = df['high'].iloc[i] - df['high'].iloc[i-1]
        if df['low'].iloc[i-1] > df['low'].iloc[i]:
            df['NDM'].iloc[i] = df['low'].iloc[i-1] - df['low'].iloc[i]

    # Calculate True Range (TR)
    df['TR'] = df[['high', 'low', 'close']].join(df['close'].shift(1)).apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))),
        axis=1
    )

    # Calculate Average True Range (ATR)
    df['ATR'] = df['TR'].ewm(span=14, adjust=False).mean()

    # Calculate Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    df['+DI'] = (df['PDM'].ewm(span=14, adjust=False).mean() / df['ATR']) * 100
    df['-DI'] = (df['NDM'].ewm(span=14, adjust=False).mean() / df['ATR']) * 100

    # Calculate Relative Strength (RS)
    df['RS'] = df['+DI'] / df['-DI']

    # Calculate Relative Strength Index (RSI)
    df['RSI'] = 100 - (100 / (1 + df['RS']))

    # Final Alpha Factor
    df['final_alpha_factor'] = df['RSI'] * df['rolling_sum_diff']

    return df['final_alpha_factor']
