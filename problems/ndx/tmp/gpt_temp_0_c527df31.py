import pandas as pd
import numpy as pd

def heuristics_v2(df):
    # Money Flow Index (MFI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    money_ratio = positive_flow.rolling(window=14).sum() / negative_flow.rolling(window=14).sum()
    mfi = 100 - (100 / (1 + money_ratio))
    mfi_ema = mfi.ewm(span=20, adjust=False).mean()
    
    # Average True Range (ATR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    
    # Square root of Closing Price to ATR Ratio
    close_to_atr_ratio_sqrt = np.sqrt(df['close'] / atr)
    
    # Composite heuristic matrix
    heuristics_matrix = mfi_ema + close_to_atr_ratio_sqrt
    
    return heuristics_matrix
