import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Momentum-Based Indicators
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_Crossover'] = df['SMA_5'] - df['SMA_20']
    
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['EMA_Crossover'] = df['EMA_10'] - df['EMA_30']
    
    # Volume-Based Indicators
    df['PVT'] = 0
    pvt = 0
    for i in range(1, len(df)):
        pvt += df.loc[df.index[i], 'volume'] * (df.loc[df.index[i], 'close'] - df.loc[df.index[i-1], 'open']) / df.loc[df.index[i-1], 'open']
        df.loc[df.index[i], 'PVT'] = pvt
    
    df['OBV'] = 0
    obv = 0
    for i in range(1, len(df)):
        if df.loc[df.index[i], 'close'] > df.loc[df.index[i-1], 'close']:
            obv += df.loc[df.index[i], 'volume']
        elif df.loc[df.index[i], 'close'] < df.loc[df.index[i-1], 'close']:
            obv -= df.loc[df.index[i], 'volume']
        df.loc[df.index[i], 'OBV'] = obv
    
    # Recent Market Behavior Focus
    df['Recent_High_Low'] = (df['high'].shift(1) - df['low'].shift(1)) - (df['high'].shift(2) - df['low'].shift(2))
    df['Average_Volume_5'] = df['volume'].rolling(window=5).mean()
    df['Significant_Volume_Change'] = (df['volume'] - df['Average_Volume_5']) / df['Average_Volume_5']
    
    # Combined Alpha Factors
    df['Hybrid_Momentum_Volume'] = df['SMA_Crossover'] * df['PVT']
    df['Hybrid_Price_Volume_Volatility'] = df['Recent_High_Low'] * df['OBV']
    
    # Advanced Alpha Factor Refinement
    df['Conditional_Factor'] = 0
    df.loc[(df['SMA_5'] > df['SMA_20']) & (df['OBV'] > df['OBV'].shift(1)), 'Conditional_Factor'] = \
        (df['SMA_5'] - df['SMA_20']) * (df['OBV'] - df['OBV'].shift(1))
    
    df['Volume_Price_Correlation'] = df['volume'].rolling(window=10).corr(df['close'])
    
    # Final Alpha Factor
    alpha_factor = df['Conditional_Factor'] + df['Volume_Price_Correlation'] + df['Hybrid_Price_Volume_Volatility']
    
    return alpha_factor
