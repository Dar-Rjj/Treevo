import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 7-day and 21-day simple moving averages (SMA) of closing prices
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['SMA_21'] = df['close'].rolling(window=21).mean()
    
    # Calculate the true range (TR) for each day
    df['TR'] = df[['high', 'low', 'close']].diff(axis=1).max(axis=1).abs()
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    # Calculate daily returns
    df['returns'] = df['close'].pct_change()
    df['volatility_10'] = df['returns'].rolling(window=10).std()
    
    # Calculate the rate of change (ROC) for 15 days
    df['ROC_15'] = df['close'].pct_change(periods=15)
    
    # Calculate the relative strength index (RSI) over 14 days
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Calculate the on-balance volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Calculate the volume weighted average price (VWAP)
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Create a composite factor using SMA, ATR, and OBV
    df['composite_factor'] = (df['SMA_7'] - df['SMA_21']) / df['ATR_14'] * df['OBV']
    
    # Form a signal by combining RSI and ROC
    df['signal'] = df['RSI_14'] * df['ROC_15']
    
    # Calculate the percentage price oscillator (PPO) using 12-day and 26-day EMAs
    EMA_12 = df['close'].ewm(span=12, adjust=False).mean()
    EMA_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['PPO'] = ((EMA_12 - EMA_26) / EMA_26) * 100
    
    # Compute the money flow index (MFI) over 14 days
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_money_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    negative_money_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    positive_mf_sum = positive_money_flow.rolling(window=14).sum()
    negative_mf_sum = negative_money_flow.rolling(window=14).sum()
    df['MFI_14'] = 100 - (100 / (1 + (positive_mf_sum / negative_mf_sum)))
    
    # Analyze the accumulation/distribution line (A/D Line)
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['volume']
    df['ADL'] = money_flow_volume.cumsum()
    
    # Calculate the intraday volatility using the highest and lowest price within a day
    df['intraday_volatility'] = (df['high'] - df['low']) / df['close']
    
    # Compute the intraday volume-weighted average price (IVWAP)
    df['IVWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Return the composite factor and other alpha factors
    return df[['composite_factor', 'signal', 'PPO', 'MFI_14', 'ADL', 'intraday_volatility', 'IVWAP']]
