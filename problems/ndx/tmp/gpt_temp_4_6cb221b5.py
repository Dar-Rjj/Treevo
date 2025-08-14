import pandas as pd
import numpy as np
def heuristics_v2(df):
    # Price Momentum
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_60'] = df['close'].ewm(span=60, adjust=False).mean()
    
    # Volume Momentum
    df['Volume_EMA_20'] = df['volume'].ewm(span=20, adjust=False).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_EMA_20']
    
    # Historical Volatility
    df['Daily_Returns'] = df['close'].pct_change()
    df['Hist_Vol_10'] = df['Daily_Returns'].rolling(window=10).std() * np.sqrt(252)
    df['Hist_Vol_30'] = df['Daily_Returns'].rolling(window=30).std() * np.sqrt(252)
    df['Hist_Vol_60'] = df['Daily_Returns'].rolling(window=60).std() * np.sqrt(252)
    df['Hist_Vol_120'] = df['Daily_Returns'].rolling(window=120).std() * np.sqrt(252)
    
    # Intraday Volatility
    df['Intraday_Vol'] = (df['high'] - df['low']) / df['open']
    df['Avg_Intraday_Vol_5'] = df['Intraday_Vol'].rolling(window=5).mean()
    df['Avg_Intraday_Vol_20'] = df['Intraday_Vol'].rolling(window=20).mean()
    
    # Trend Strength Indicators
    # ADX calculation (without normalization)
    def adx(high, low, close, period=14):
        tr = pd.DataFrame(index=close.index)
        tr['H-L'] = abs(high - low)
        tr['H-PC'] = abs(high - close.shift(1))
        tr['L-PC'] = abs(low - close.shift(1))
        tr['TR'] = tr.max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=close.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=close.index)
        
        atr = tr['TR'].rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        
        return adx
