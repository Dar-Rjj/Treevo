import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume and Amount Based Factors
    df['SMA_5_Volume'] = df['volume'].rolling(window=5).mean()
    df['SMA_20_Volume'] = df['volume'].rolling(window=20).mean()
    df['Ratio_SMA_Volume'] = df['SMA_5_Volume'] / df['SMA_20_Volume']
    
    df['Max_Vol_10D'] = df['volume'].rolling(window=10).max()
    df['Vol_Breakout'] = df['volume'] / df['Max_Vol_10D']
    
    df['Vol_Corr'] = df['volume'].rolling(window=30).corr(df['close'])
    
    # Price Trend and Momentum Indicators
    df['EMA_12_Close'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26_Close'] = df['close'].ewm(span=26, adjust=False).mean()
    df['EMA_Ratio'] = df['EMA_12_Close'] / df['EMA_26_Close']
    
    df['Max_Price_20D'] = df['high'].rolling(window=20).max()
    df['Min_Price_20D'] = df['low'].rolling(window=20).min()
    df['Price_Breakout_Up'] = (df['high'] - df['Max_Price_20D']) / df['Max_Price_20D']
    df['Price_Breakout_Down'] = (df['Min_Price_20D'] - df['low']) / df['Min_Price_20D']
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # True Range and Average True Range (ATR)
    df['TR'] = df[['high', 'close']].apply(lambda x: max(x[0] - df['low'], abs(x[0] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))), axis=1)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    # Historical Volatility (HV)
    log_returns = np.log(df['close'] / df['close'].shift(1))
    df['HV_10D'] = log_returns.rolling(window=10).std() * np.sqrt(252)
    
    # Composite Factors
    df['CMI'] = (df['EMA_Ratio'] + df['Price_Breakout_Up'] - df['Price_Breakout_Down'] + df['Vol_Breakout']) / 4
    df['IAS'] = (df['CMI'] * df['RSI_14']) + (df['ATR_14'] * df['HV_10D']) - df['Vol_Corr']
    
    return df['IAS']
