import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate EMAs
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Compute the difference between EMAs of different lengths
    df['EMA_diff_5_20'] = df['EMA_5'] - df['EMA_20']
    
    # Calculate the rate of change (ROC) for each EMA
    df['ROC_EMA_5'] = df['EMA_5'].pct_change()
    df['ROC_EMA_20'] = df['EMA_20'].pct_change()
    
    # Price Oscillators
    df['High_Low_Diff'] = df['high'] - df['low']
    df['Close_Open_Pct_Change'] = (df['close'] / df['open']) - 1
    df['Close_EMA50_Ratio'] = df['close'] / df['close'].ewm(span=50, adjust=False).mean()
    
    # Volume Patterns
    df['Avg_Volume_5'] = df['volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['volume'] / df['Avg_Volume_5']
    df['Volume_Spike'] = (df['volume'] > df['Avg_Volume_5'] * 1.5).astype(int)
    
    # Advanced Candlestick Patterns
    df['Bullish_Engulfing'] = ((df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & 
                               (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))).astype(int)
    df['Bearish_Engulfing'] = ((df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & 
                               (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))).astype(int)
    df['Doji'] = ((df['close'] - df['open']).abs() < (df['high'] - df['low']) * 0.05).astype(int)
    df['Hammer'] = ((df['close'] - df['low']) > (0.6 * (df['high'] - df['low'])) & 
                    ((df['open'] - df['low']) > (0.6 * (df['high'] - df['low'])))).astype(int)
    df['Hanging_Man'] = ((df['high'] - df['close']) > (0.6 * (df['high'] - df['low'])) & 
                         ((df['high'] - df['open']) > (0.6 * (df['high'] - df['low'])))).astype(int)
    df['Morning_Star'] = ((df['close'].shift(2) < df['open'].shift(2)) & 
                          (df['close'].shift(1) < df['open'].shift(1)) & 
                          (df['close'] > df['open']) & 
                          (df['close'] > df['open'].shift(2))).astype(int)
    df['Evening_Star'] = ((df['close'].shift(2) > df['open'].shift(2)) & 
                          (df['close'].shift(1) > df['open'].shift(1)) & 
                          (df['close'] < df['open']) & 
                          (df['close'] < df['open'].shift(2))).astype(int)
    
    # Combine Multiple Indicators
    df['Composite_Score'] = (df['EMA_diff_5_20'] + df['Close_Open_Pct_Change'] + df['Volume_Ratio'] + 
                             df['Bullish_Engulfing'] - df['Bearish_Engulfing'] + df['Doji'] + 
                             df['Hammer'] - df['Hanging_Man'] + df['Morning_Star'] - df['Evening_Star'])
    
    return df['Composite_Score']
