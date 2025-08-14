import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the 50-day and 200-day Simple Moving Averages (SMA)
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    # Generate Buy/Sell Signal When Short-Term SMA Crosses Above/Below Long-Term SMA
    df['SMA_Crossover'] = (df['SMA_50'] > df['SMA_200']).astype(int) - (df['SMA_50'] < df['SMA_200']).astype(int)
    
    # Calculate RSI Using Close Prices
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Identify Overbought (RSI > 70) and Oversold (RSI < 30) Conditions
    df['RSI_Signal'] = ((df['RSI'] < 30).astype(int) - (df['RSI'] > 70).astype(int))
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = 0
    obv = 0
    for i in range(1, len(df)):
        if df['close'][i] > df['close'][i-1]:
            obv += df['volume'][i]
        elif df['close'][i] < df['close'][i-1]:
            obv -= df['volume'][i]
        else:
            obv += 0
        df.loc[df.index[i], 'OBV'] = obv
    
    # Use OBV to Confirm Trend Reversals
    df['OBV_Trend'] = df['OBV'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # Develop a Composite Score
    df['Composite_Score'] = (df['SMA_Crossover'] * 0.4) + (df['RSI_Signal'] * 0.3) + (df['OBV_Trend'] * 0.3)
    
    return df['Composite_Score']
