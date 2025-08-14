import numpy as np
def heuristics_v2(df):
    # Base Factors
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_60'] = df['close'].rolling(window=60).mean()
    
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_60'] = df['close'].ewm(span=60, adjust=False).mean()
    
    df['ROC_Daily'] = df['close'].pct_change()
    df['ROC_Weekly'] = df['close'].pct_change(periods=5)
    df['ROC_Monthly'] = df['close'].pct_change(periods=21)
    
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['VM'] = df['volume'].pct_change()
    
    df['BAS'] = df['high'] - df['low']
    df['VT'] = df['volume'].pct_change()
    df['DV'] = df['close'] * df['volume']
    
    df['HV'] = df['close'].rolling(window=20).std()
    df['ATR'] = df[['high' - 'low', abs('high' - df['close'].shift()), abs('low' - df['close'].shift())]].max(axis=1).rolling(window=14).mean()
    df['BB_Upper'] = df['SMA_20'] + 2 * df['close'].rolling(window=20).std()
    df['BB_Lower'] = df['SMA_20'] - 2 * df['close'].rolling(window=20).std()
    
    # Composite Indicators
    df['Momentum_Volatility_Factor'] = df['ROC_Daily'] / df['HV']
    df['Volume_Momentum_Factor'] = df['VM'] * df['OBV']
    df['Liquidity_Momentum_Factor'] = df['BAS'] * df['ROC_Daily']
    
    # Adaptive Lookback Periods
    def variable_lookback_factor(series, short_window, medium_window, long_window):
        return series.ewm(span=short_window, adjust=False).mean() + series.ewm(span=medium_window, adjust=False).mean() + series.ewm(span=long_window, adjust=False).mean()
