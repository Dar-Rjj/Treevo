import numpy as np
def heuristics_v2(df):
    # Calculate Simple Moving Average (SMA) of Close Prices Over 20 Days
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # Calculate Exponential Moving Average (EMA) of Close Prices Over 12 Days
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    
    # Calculate Relative Strength Index (RSI) on Close Prices Over Last 14 Days
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Calculate Rate of Change (ROC) Indicator Over Last 9 Days
    df['ROC_9'] = df['close'].pct_change(periods=9)
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Calculate Volume Moving Average (VMA) Over Last 20 Days
    df['VMA_20'] = df['volume'].rolling(window=20).mean()
    
    # Calculate True Range
    df['TR'] = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
    
    # Calculate Average True Range (ATR) Over Last 14 Days
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    # Calculate Donchian Channels
    df['Donchian_Upper'] = df['high'].rolling(window=20).max()
    df['Donchian_Lower'] = df['low'].rolling(window=20).min()
    
    # Calculate Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    df['BB_STD'] = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_STD']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_STD']
    
    # Calculate Historical Volatility (HV) Over Last 20 Days
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
