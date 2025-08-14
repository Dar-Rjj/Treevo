import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Close Price Change from Previous Day
    df['Close_Change'] = df['close'] - df['close'].shift(1)
    
    # Calculate True Range (TR)
    df['TR'] = df[['high' - 'low',
                   ('high' - df['close'].shift(1)).abs(),
                   ('low' - df['close'].shift(1)).abs()]].max(axis=1)
    
    # Calculate Average True Range (ATR) over 14 days
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    # Calculate +DM and -DM
    df['+DM'] = (df['high'].diff().where(lambda x: x > 0, 0) - 
                 df['low'].shift(1).diff().clip(lower=0))
    df['-DM'] = (df['low'].shift(1).diff().where(lambda x: x > 0, 0) -
                 df['high'].diff().clip(lower=0))
    
    # Smooth +DM and -DM over 14 days
    df['+DM_14'] = df['+DM'].rolling(window=14).mean()
    df['-DM_14'] = df['-DM'].rolling(window=14).mean()
    
    # Convert smoothed +DM and -DM to DI+ and DI-
    df['+DI'] = 100 * (df['+DM_14'] / df['ATR_14'])
    df['-DI'] = 100 * (df['-DM_14'] / df['ATR_14'])
    
    # Derive DMI difference
    df['DMI_Diff'] = df['+DI'] - df['-DI']
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = (df['volume'] * ((df['close'] > df['close'].shift(1)) - (df['close'] < df['close'].shift(1))).cumsum())
    
    # Calculate Price-Volume Trend (PVT)
    df['PVT'] = (df['volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)).cumsum()
    
    # Calculate Elder's Force Index
    df['ForceIndex'] = (df['close'] - df['close'].shift(1)) * df['volume']
    
    # Calculate Typical Price (TP)
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate Raw Money Flow
    df['Raw_Money_Flow'] = df['TP'] * df['volume']
    
    # Determine Positive and Negative Money Flow
    df['Positive_MF'] = df['Raw_Money_Flow'].where(df['TP'] > df['TP'].shift(1), 0)
    df['Negative_MF'] = -df['Raw_Money_Flow'].where(df['TP'] < df['TP'].shift(1), 0)
    
    # Calculate the Money Flow Ratio
    Positive_MF_14 = df['Positive_MF'].rolling(window=14).sum()
    Negative_MF_14 = df['Negative_MF'].rolling(window=14).sum()
    df['Money_Flow_Ratio'] = Positive_MF_14 / Negative_MF_14
    
    # Calculate Money Flow Index (MFI)
    df['MFI'] = 100 - (100 / (1 + df['Money_Flow_Ratio']))
    
    # Return a novel factor combining several indicators
    return df['Close_Change'] + df['ATR_14'] + df['DMI_Diff'] + df['OBV'] + df['PVT'] + df['ForceIndex'] + df['MFI']

# Example usage:
# df = pd.DataFrame(...)  # Load your data
# factors = heuristics_v2(df)
