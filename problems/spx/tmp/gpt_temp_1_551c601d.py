import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=14):
    # Thought 2: Simple Price Movement over N Days
    df['Price_Movement'] = (df['close'] / df['close'].shift(n)) - 1
    
    # Thought 3: High-Low Range Expansion
    df['High_Low_Range'] = (df['high'] - df['low']) / df['close']
    
    # Thought 10: Average True Range
    df['True_Range'] = df[['high', 'low']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['close'].shift(1)), abs(x[1] - df['close'].shift(1))), axis=1)
    df['ATR'] = df['True_Range'].rolling(window=n).mean()
    
    # Thought 5: Volume-Weighted Price
    vwp_numerator = (df['close'] * df['volume']).rolling(window=n).sum()
    vwp_denominator = df['volume'].rolling(window=n).sum()
    df['Volume_Weighted_Price'] = vwp_numerator / vwp_denominator
    
    # Thought 6: Change in Volume-Weighted Price
    df['VWP_Change'] = (df['Volume_Weighted_Price'] / df['Volume_Weighted_Price'].shift(1)) - 1
    
    # Thought 11: Volume-Weighted Moving Average
    df['VWMA'] = (df['close'] * df['volume']).rolling(window=n).sum() / df['volume'].rolling(window=n).sum()
    
    # Thought 8: Money Flow Index (MFI)
    df['Typical_Price'] = (df['high'] + df['low'] + df['close']) / 3
    df['Raw_Money_Flow'] = df['Typical_Price'] * df['volume']
    df['Positive_Money_Flow'] = df[df['Typical_Price'] > df['Typical_Price'].shift(1)]['Raw_Money_Flow'].rolling(window=n).sum()
    df['Negative_Money_Flow'] = df[df['Typical_Price'] < df['Typical_Price'].shift(1)]['Raw_Money_Flow'].rolling(window=n).sum()
    df['MFI'] = 100 - (100 / (1 + (df['Positive_Money_Flow'] / df['Negative_Money_Flow'])))
    
    # Thought 9: Chaikin Oscillator
    df['A_D_Line'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    df['A_D_Line_3day_EMA'] = df['A_D_Line'].ewm(span=3, adjust=False).mean()
    df['A_D_Line_10day_EMA'] = df['A_D_Line'].ewm(span=10, adjust=False).mean()
    df['Chaikin_Oscillator'] = df['A_D_Line_3day_EMA'] - df['A_D_Line_10day_EMA']
    
    # Thought 12: On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    
    # Combine all factors into a single factor
    df['Combined_Factor'] = (df['Price_Movement'] + df['High_Low_Range'] + df['ATR'] + 
                             df['VWP_Change'] + df['VWMA'] + df['MFI'] + df['Chaikin_Oscillator'] + df['OBV'])
    
    return df['Combined_Factor']
