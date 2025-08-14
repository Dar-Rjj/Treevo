import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate EMAs
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Crossover Signal
    df['Crossover_Signal'] = df['EMA_50'] - df['EMA_200']
    
    # Price Change Ratio
    df['Price_Change_Ratio'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # VWAP
    df['Daily_VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Volume Relative Strength Index (VRSI)
    df['UpVol'] = np.where(df['volume'] > df['volume'].shift(1), df['volume'], 0)
    df['DownVol'] = np.where(df['volume'] < df['volume'].shift(1), df['volume'], 0)
    df['AvgUpVol'] = df['UpVol'].ewm(span=14, adjust=False).mean()
    df['AvgDownVol'] = df['DownVol'].ewm(span=14, adjust=False).mean()
    df['VRSI'] = df['AvgUpVol'] / (df['AvgUpVol'] + df['AvgDownVol'])
    
    # Volume Increase Indicator
    df['Volume_Increase'] = np.where(df['volume'] > df['volume'].shift(1), 1, 0)
    
    # Bullish Engulfing Pattern
    df['Bullish_Pattern_Condition'] = (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
    df['Bullish_Pattern_Confirmation'] = df['Bullish_Pattern_Condition'] & (df['close'] > df['open'])
    
    # Bearish Engulfing Pattern
    df['Bearish_Pattern_Condition'] = (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))
    df['Bearish_Pattern_Confirmation'] = df['Bearish_Pattern_Condition'] & (df['close'] < df['open'])
    
    # Hammer Pattern
    df['Hammer_Body'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['Hammer_Shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low'])
    df['Hammer_Condition'] = (df['Hammer_Body'] <= 0.3) & (df['Hammer_Shadow'] >= 0.6)
    df['Hammer_Confirmation'] = df['Hammer_Condition'] & (df['close'] > df['open'])
    
    # Intraday Movement Analysis
    df['Intraday_High_Low_Spread'] = df['high'] - df['low']
    df['Location_Ratio'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['Intraday_Volatility'] = (df['high'] - df['low']) / df['close']
    
    # Combined Alpha Factor
    df['Alpha_Factor'] = (0.3 * df['Crossover_Signal']) + \
                         (0.2 * df['VRSI']) + \
                         (0.1 * (df['Bullish_Pattern_Confirmation'] - df['Bearish_Pattern_Confirmation'] + df['Hammer_Confirmation'])) + \
                         (0.2 * df['Price_Change_Ratio']) + \
                         (0.2 * (df['Intraday_High_Low_Spread'] * df['Location_Ratio'] * df['Intraday_Volatility']))
    
    return df['Alpha_Factor']
