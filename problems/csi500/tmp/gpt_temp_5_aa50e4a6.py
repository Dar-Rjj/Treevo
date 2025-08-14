import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate the 5-day and 20-day Simple Moving Averages (SMA)
    df['5_day_SMA'] = df['close'].rolling(window=5).mean()
    df['20_day_SMA'] = df['close'].rolling(window=20).mean()
    
    # Compute the SMA Crossover Signal
    df['SMA_Crossover_Signal'] = (df['5_day_SMA'] > df['20_day_SMA']).astype(int)
    
    # Calculate the True Range for each day
    df['True_Range'] = df[['high', 'low']].apply(lambda x: max(x['high'], x.shift(1)['close']) - min(x['low'], x.shift(1)['close']), axis=1)
    
    # Calculate the 14-day Average True Range (ATR)
    df['14_day_ATR'] = df['True_Range'].rolling(window=14).mean()
    
    # Define a threshold for high and low ATR (e.g., 75th percentile for high, 25th percentile for low)
    atr_threshold_high = df['14_day_ATR'].quantile(0.75)
    atr_threshold_low = df['14_day_ATR'].quantile(0.25)
    
    # Combine factors into a composite score
    conditions = [
        (df['SMA_Crossover_Signal'] == 1) & (df['14_day_ATR'] > atr_threshold_high),
        (df['SMA_Crossover_Signal'] == 1) & (df['14_day_ATR'] <= atr_threshold_high),
        (df['SMA_Crossover_Signal'] == 0)
    ]
    choices = [1.0, 0.5, 0.0]
    
    df['Composite_Factor'] = pd.np.select(conditions, choices, default=0.0)
    
    return df['Composite_Factor']
