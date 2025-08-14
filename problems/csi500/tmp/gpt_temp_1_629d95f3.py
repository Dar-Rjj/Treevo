import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Total Volume and Total Dollar Value for each day to get VWAP
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['close'] * df['volume']
    
    # Group by date to aggregate daily data
    daily_data = df.groupby(df.index.date).agg({'total_dollar_value': 'sum', 'total_volume': 'sum'})
    
    # Calculate Daily VWAP
    daily_data['vwap'] = daily_data['total_dollar_value'] / daily_data['total_volume']
    
    # Merge VWAP back into the original dataframe
    df = df.join(daily_data['vwap'], on=df.index.date)
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df.groupby(df.index.date)['vwap_deviation'].cumsum()
    
    # Final VWAP Cumulative Deviation Factor (we can use the last value of cumulative deviation per day)
    vwap_cumulative_deviation_factor = df.groupby(df.index.date)['cumulative_vwap_deviation'].last()
    
    return vwap_cumulative_deviation_factor
