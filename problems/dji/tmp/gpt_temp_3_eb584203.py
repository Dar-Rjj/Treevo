import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum Indicators
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    df['SMA_Ratio'] = df['SMA_50'] / df['SMA_200']
    
    df['ROC_14'] = df['close'].pct_change(periods=14)
    df['ROC_28'] = df['close'].pct_change(periods=28)
    
    # Volatility Indicators
    df['TR'] = df[['high', 'low', 'close']].apply(lambda x: max(x) - min(x), axis=1)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
    df['SD_20'] = df['Log_Returns'].rolling(window=20).std()
    df['SD_60'] = df['Log_Returns'].rolling(window=60).std()
    
    # Volume Indicators
    df['OBV'] = (df['close'] > df['close'].shift(1)).astype(int) * df['volume'] - (df['close'] < df['close'].shift(1)).astype(int) * df['volume']
    df['OBV'] = df['OBV'].cumsum()
    
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['volume']
    df['CMF_20'] = money_flow_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Price Patterns
    df['Doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1).astype(int)
    
    # Technical Analysis Combinations
    df['Momentum_Volatility_Ratio'] = df['ROC_14'] / df['ATR_14']
    
    df['Price_Volume_Trend'] = df['Log_Returns'].rolling(window=20).corr(df['OBV'].pct_change().rolling(window=20))
    
    df['Composite_Indicator'] = df['SMA_Ratio'] + df['ATR_14'] + df['OBV'].pct_change(periods=20)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df['Composite_Indicator']
