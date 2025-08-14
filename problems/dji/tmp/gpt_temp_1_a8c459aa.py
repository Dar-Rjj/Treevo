import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=14):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = df['close'].rolling(window=n).mean()
    
    # Calculate Exponential Moving Average (EMA)
    df['EMA'] = df['close'].ewm(span=n, adjust=False).mean()
    
    # Calculate True Range (TR)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate Average True Range (ATR)
    df['ATR'] = df['TR'].rolling(window=n).mean()
    
    # Calculate Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=n).mean()
    avg_loss = loss.rolling(window=n).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Calculate Moving Average Convergence Divergence (MACD)
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_line'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
    
    # Calculate Chaikin Oscillator
    df['money_flow_multiplier'] = (df['close'] - df['low']) - (df['high'] - df['close']) / (df['high'] - df['low'])
    df['money_flow_volume'] = df['money_flow_multiplier'] * df['volume']
    df['ADL'] = df['money_flow_volume'].cumsum()
    df['Chaikin_Oscillator'] = df['ADL'].ewm(span=3, adjust=False).mean() - df['ADL'].ewm(span=10, adjust=False).mean()
    
    # Calculate Money Flow Index (MFI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    positive_flow_sum = positive_flow.rolling(window=n).sum()
    negative_flow_sum = negative_flow.rolling(window=n).sum()
    money_ratio = positive_flow_sum / negative_flow_sum
    df['MFI'] = 100 - (100 / (1 + money_ratio))
    
    # Identify Inside Bar and Outside Bar
    df['Inside_Bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    df['Outside_Bar'] = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
    
    # Calculate Volume Ratio
    df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(window=n).mean()
    
    # Calculate Volume Price Trend (VPT)
    df['VPT'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
    
    # Combine factors
    df['Factor_1'] = df['SMA'] + df['ATR'] + df['OBV']
    df['Factor_2'] = df['RSI'] + df['MACD_line'] + df['MFI']
    df['Factor_3'] = df['Inside_Bar'].astype(int) + df['Outside_Bar'].astype(int) + df['VPT']
    
    return df[['Factor_1', 'Factor_2', 'Factor_3']]
