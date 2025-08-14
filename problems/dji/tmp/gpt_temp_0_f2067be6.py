import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Momentum Indicators
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ROC_12'] = (df['close'] - df['close'].shift(12)) / df['close'].shift(12) * 100
    
    # Volume Indicators
    df['OBV'] = (df['close'] > df['close'].shift(1)).astype(int) * df['volume']
    df['OBV'] = df['OBV'].cumsum()
    
    # Volatility Indicator
    df['TR'] = df[['high' - 'low', 'high' - df['close'].shift(1), 
                   df['low'] - df['close'].shift(1)]].max(axis=1)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    # Price Patterns
    df['Breakout'] = (df['high'] > df['high'].rolling(window=20).max()).astype(int)
    df['Reversal'] = ((df['open'] > df['close']) & (df['open'].shift(1) < df['close'].shift(1))).astype(int)
    
    # Trend Indicators
    df['+DM'] = df['high'].diff().where(lambda x: x > 0, 0)
    df['-DM'] = -df['low'].diff().where(lambda x: x < 0, 0)
    df['+DI'] = 100 * (df['+DM'].rolling(window=14).sum() / df['TR'].rolling(window=14).sum())
    df['-DI'] = 100 * (df['-DM'].rolling(window=14).sum() / df['TR'].rolling(window=14).sum())
    
    # Composite Factors
    df['SMA_OBV'] = df['SMA_50'] + df['OBV']
    df['Volatility_Trend'] = df['ATR_14'] * (df['+DI'] - df['-DI'])
    
    # Custom Indicators
    df['HLDiff'] = df['high'] - df['low']
    df['OCRatio'] = df['open'] / df['close']
    df['VWP'] = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
    
    # Interpretable Alpha Factors
    df['Momentum_Strength'] = df['SMA_50'] - df['EMA_20']
    df['Trend_Confirmation'] = df['+DI'] - df['-DI']
    df['Volume_Price_Consistency'] = df['VWP'].rolling(window=20).corr(df['close'])
    df['Breakout_Opportunity'] = df.groupby((df['Breakout'] == 0).cumsum())['Breakout'].cumcount()
    
    return df[['Momentum_Strength', 'Trend_Confirmation', 'Volume_Price_Consistency', 'Breakout_Opportunity']]
