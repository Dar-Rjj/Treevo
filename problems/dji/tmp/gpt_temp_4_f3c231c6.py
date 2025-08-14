import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics(df):
    # Calculate Intraday Price Range
    intraday_range = df['high'] - df['low']
    
    # Adjust for Volume
    volume_adjustment = (df['volume'] / df['volume'].rolling(window=5).mean()).fillna(1)
    
    # Combine Intraday Range and Volume Adjustment
    combined_intraday = intraday_range * volume_adjustment
    
    # Smoothing with Exponential Moving Average (EMA) with Span of 5
    smoothed_intraday = combined_intraday.ewm(span=5, adjust=False).mean()
    
    # Calculate MACD
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    # Determine Momentum
    momentum = np.where(macd_line > signal_line, 1, -1)
    
    # Measure Volume Synchronization with Shock Filter
    volume_change = df['volume'] - df['volume'].shift(1)
    price_change = df['close'] - df['close'].shift(1)
    volume_ratio = df['volume'] / df['volume'].shift(1)
    volume_shock_threshold = 1.5
    shock_filter = np.where(volume_ratio > volume_shock_threshold, 1, 0)
    synchronized_product = volume_change * price_change * shock_filter
    synchronized_sign = np.sign(synchronized_product)
    
    # Calculate On-Balance Volume (OBV)
    obv = pd.Series(index=df.index, dtype='float64')
    obv[0] = 0
    obv[1:] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Weight OBV by |MACD Line - Signal Line|
    weight = np.abs(macd_line - signal_line)
    weighted_obv = obv * weight
    
    # Final Alpha Factor
    final_alpha_factor = smoothed_intraday + synchronized_sign + weighted_obv
    
    return final_alpha_factor
