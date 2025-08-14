import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily gain or loss
    df['Daily_Gain_Loss'] = df['close'].diff()
    
    # Assign sign: positive for gain, negative for loss
    df['Daily_Gain_Loss_Signed'] = np.sign(df['Daily_Gain_Loss']) * df['Daily_Gain_Loss']
    
    # Adjust gain/loss by volume
    df['Adjusted_Gain_Loss'] = df['Daily_Gain_Loss_Signed'] * df['volume']
    
    # Calculate price volatility using Average True Range (ATR)
    df['True_Range'] = df[['high', 'low']].apply(lambda x: max(x['high'], df['close'].shift(1)) - min(x['low'], df['close'].shift(1)), axis=1)
    df['Average_True_Range'] = df['True_Range'].rolling(window=5).mean()
    
    # Adjust gain/loss by price volatility
    df['Price_Adjusted_Gain_Loss'] = df['Adjusted_Gain_Loss'] / df['Average_True_Range']
    
    # Cumulate adjusted value over a window of 5 days
    df['Cumulative_Adjusted_Value'] = df['Price_Adjusted_Gain_Loss'].rolling(window=5).sum()
    
    # Output the final metric for each day
    return df['Cumulative_Adjusted_Value']
