import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum Indicators
    df['50_day_SMA'] = df['close'].rolling(window=50).mean()
    df['200_day_SMA'] = df['close'].rolling(window=200).mean()
    df['SMA_ratio'] = df['50_day_SMA'] / df['200_day_SMA']
    
    df['14_day_ROC'] = df['close'].pct_change(periods=14)
    df['28_day_ROC'] = df['close'].pct_change(periods=28)
    
    # Volatility Indicators
    df['true_range'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
    df['14_day_ATR'] = df['true_range'].rolling(window=14).mean()
    
    df['log_returns'] = np.log(df['close']) - np.log(df['close'].shift(1))
    df['20_day_SD'] = df['log_returns'].rolling(window=20).std()
    df['60_day_SD'] = df['log_returns'].rolling(window=60).std()
    
    # Volume Indicators
    df['OBV'] = (df['close'] > df['close'].shift(1)).astype(int) * df['volume'] - (df['close'] < df['close'].shift(1)).astype(int) * df['volume']
    df['OBV'] = df['OBV'].cumsum()
    
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['volume']
    df['20_day_CMF'] = money_flow_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Technical Analysis Combinations
    df['momentum_volatility_ratio'] = df['14_day_ROC'] / df['14_day_ATR']
    
    df['20_day_log_returns'] = df['log_returns'].rolling(window=20).sum()
    df['price_volume_trend'] = df['20_day_log_returns'].rolling(window=20).corr(df['OBV'].rolling(window=20))
    
    # Composite Indicator
    df['composite_indicator'] = (df['SMA_ratio'] + df['14_day_ATR'] + df['OBV']).rolling(window=20).mean()
    
    return df['composite_indicator']
