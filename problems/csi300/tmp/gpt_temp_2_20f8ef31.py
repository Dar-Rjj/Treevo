import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum Indicators
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    df['ROC_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100
    df['ROC_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
    df['ROC_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100
    
    df['Momentum_10'] = df['close'] - df['close'].shift(10)
    
    # Volume Trends
    df['Volume_SMA_30'] = df['volume'].rolling(window=30).mean()
    df['Volume_Above_SMA_30'] = df['volume'] > df['Volume_SMA_30']
    
    df['VPT'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
    df['VPT'] = df['VPT'].cumsum()
    
    # Volatility Measures
    df['Volatility_20'] = df['close'].rolling(window=20).std()
    
    df['True_Range'] = df[['high' - 'low', (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()]].max(axis=1)
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()
    
    # Market Sentiment Analysis
    df['Bullish_Signal'] = df['close'] > df['open']
    df['Bearish_Signal'] = df['close'] < df['open']
    
    df['ADL'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    df['ADL'] = df['ADL'].cumsum()
    
    # Inter-day Price Movement Analysis
    df['Gap_Up'] = (df['open'] - df['close'].shift(1)) > 0.01 * df['close'].shift(1)
    df['Gap_Down'] = (df['open'] - df['close'].shift(1)) < -0.01 * df['close'].shift(1)
    
    # Composite Indicator
    df['Composite_Score'] = (df['SMA_5'] + df['SMA_10'] + df['SMA_20'] + df['ROC_5'] + df['ROC_10'] + df['ROC_20'] + df['Momentum_10'] + df['VPT'] + df['ATR_14'] + df['ADL']) / 10
    
    # Advanced Pattern Recognition
    df['Recent_High'] = df['high'].rolling(window=20).max()
    df['Recent_Low'] = df['low'].rolling(window=20).min()
    
    # Trading Volume Impact on Returns
    df['Volume_Spike'] = df['volume'] > (df['volume'].rolling(window=30).mean() + 2 * df['volume'].rolling(window=30).std())
    
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # Final Alpha Factor
    alpha_factor = df['Composite_Score']
    
    return alpha_factor
