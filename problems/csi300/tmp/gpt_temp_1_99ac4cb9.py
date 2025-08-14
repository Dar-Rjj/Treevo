import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=10):
    # Calculate Intraday Reversal
    df['intraday_high_close_diff'] = df['high'] - df['close']
    df['intraday_low_open_diff'] = df['low'] - df['open']
    
    # Calculate Trading Activity Concentration
    # Assume df has a 'volume' column with volume data for each 10-minute interval
    # Resample the volume data into 10-minute intervals if not already done
    df_volume = df[['volume']].resample('10T').sum().fillna(0)
    
    # Identify Peak and Off-Peak Volumes
    max_volume = df_volume.rolling(window=6*24, min_periods=1).max()
    high_volume_intervals = (df_volume == max_volume)
    
    # Quantify Volume Concentration
    volume_concentration = high_volume_intervals.sum(axis=1) / high_volume_intervals.shape[1]
    
    # Combine Intraday Reversal and Volume Concentration
    intraday_reversal = (df['intraday_high_close_diff'] + df['intraday_low_open_diff']).abs()
    normalized_reversal = (intraday_reversal - intraday_reversal.min()) / (intraday_reversal.max() - intraday_reversal.min())
    
    combined_factor = normalized_reversal * volume_concentration
    
    # Calculate Momentum Indicator
    ema_factor = combined_factor.ewm(span=n, adjust=False).mean()
    
    return ema_factor
