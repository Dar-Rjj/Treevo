import pandas as pd
import pandas as pd

def heuristics_v2(df):
    """
    Calculate a novel and interpretable alpha factor based on the provided DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume'] and index (date).
    
    Returns:
    pd.Series: A series of factor values indexed by date.
    """
    # Calculate the daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate the 10-day moving average of the close price
    df['ma_10'] = df['close'].rolling(window=10).mean()
    
    # Calculate the 30-day moving average of the close price
    df['ma_30'] = df['close'].rolling(window=30).mean()
    
    # Calculate the difference between the 10-day and 30-day moving averages
    df['ma_diff'] = df['ma_10'] - df['ma_30']
    
    # Calculate the average true range (ATR) over 14 days
    df['tr'] = df[['high' - 'low', 
                   abs('high' - df['close'].shift(1)), 
                   abs('low' - df['close'].shift(1))]].max(axis=1)
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    
    # Calculate the ratio of the current day's volume to the 14-day average volume
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(window=14).mean()
    
    # Calculate the factor value
    df['factor'] = (df['ma_diff'] / df['atr_14']) * df['vol_ratio']
    
    # Return the factor values
    return df['factor'].dropna()
