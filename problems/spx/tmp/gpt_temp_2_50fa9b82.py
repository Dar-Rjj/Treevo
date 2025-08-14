import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Simple Moving Averages
    short_sma = df['close'].rolling(window=10).mean()
    long_sma = df['close'].rolling(window=50).mean()
    
    # SMA Crossover
    sma_crossover = short_sma - long_sma
    
    # Rate of Change (ROC)
    roc = df['close'].pct_change(periods=10) * 100
    
    # True Range
    true_range = df[['high', 'low']].apply(lambda x: max(x[0], df['close'].shift(1)) - min(x[1], df['close'].shift(1)), axis=1)
    
    # Average True Range (ATR)
    atr = true_range.rolling(window=14).mean()
    
    # Volume Weighted Moving Average (VWAP)
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Volume ROC
    volume_roc = df['volume'].pct_change(periods=10) * 100
    
    # On-Balance Volume (OBV)
    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    
    # Chaikin Money Flow (CMF)
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mf_volume = mf_multiplier * df['volume']
    cmf = mf_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Price-Volume Trend (PVT)
    pvt = (df['close'].pct_change() * df['volume']).cumsum()
    
    # Opening Price Gap
    open_gap = df['open'] - df['close'].shift(1)
    
    # Closing Price Strength
    closing_strength = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Autocorrelation
    autocorr = df['close'].rolling(window=10).apply(lambda x: x.autocorr(lag=1), raw=False)
    
    # Combine all factors into a single alpha factor
    alpha_factor = (
        sma_crossover * 0.1 +
        roc * 0.1 +
        atr * 0.1 +
        vwap * 0.1 +
        volume_roc * 0.1 +
        obv * 0.1 +
        cmf * 0.1 +
        pvt * 0.1 +
        open_gap * 0.1 +
        closing_strength * 0.1 +
        autocorr * 0.1
    )
    
    return alpha_factor
