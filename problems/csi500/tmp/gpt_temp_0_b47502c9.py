import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['Total_Dollar_Value'] = df['close'] * df['volume']
    df['VWAP'] = df.groupby(df.index.date)['Total_Dollar_Value'].transform('sum') / df.groupby(df.index.date)['volume'].transform('sum')
    
    # Calculate VWAP Deviation
    df['VWAP_Deviation'] = df['close'] - df['VWAP']
    
    # Calculate Cumulative VWAP Deviation
    df['Cumulative_VWAP_Deviation'] = df['VWAP_Deviation'].cumsum()
    
    # Integrate Short-Term Momentum (5 days)
    df['Short_Term_Momentum'] = df['VWAP_Deviation'].rolling(window=5).sum()
    
    # Integrate Medium-Term Momentum (10 days)
    df['Medium_Term_Momentum'] = df['VWAP_Deviation'].rolling(window=10).sum()
    
    # Integrate Long-Term Momentum (20 days)
    df['Long_Term_Momentum'] = df['VWAP_Deviation'].rolling(window=20).sum()
    
    # Calculate Intraday Volatility
    df['High_Low_Range'] = df['high'] - df['low']
    df['Absolute_VWAP_Deviation'] = df['VWAP_Deviation'].abs()
    df['Intraday_Volatility'] = df['High_Low_Range'] + df['Absolute_VWAP_Deviation']
    
    # Final Alpha Factor
    df['Alpha_Factor'] = (0.4 * df['Cumulative_VWAP_Deviation'] +
                          0.2 * df['Short_Term_Momentum'] +
                          0.2 * df['Medium_Term_Momentum'] +
                          0.2 * df['Long_Term_Momentum'] +
                          0.2 * df['Intraday_Volatility'])
    
    return df['Alpha_Factor']
