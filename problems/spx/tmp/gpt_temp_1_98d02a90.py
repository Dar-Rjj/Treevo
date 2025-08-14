import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from talib import ADX, MFI

def heuristics_v2(df):
    # Calculate 5-day and 20-day simple moving averages (SMA) for the closing prices
    df['5_day_SMA'] = df['close'].rolling(window=5).mean()
    df['20_day_SMA'] = df['close'].rolling(window=20).mean()
    
    # Alpha factor: Ratio of the 5-day SMA to the 20-day SMA
    df['sma_ratio'] = df['5_day_SMA'] / df['20_day_SMA']
    
    # 20-day price return
    df['20_day_price_return'] = df['close'].pct_change(periods=20)
    
    # 10-day Average Directional Index (ADX)
    df['ADX'] = ADX(df['high'], df['low'], df['close'], timeperiod=10)
    
    # 20-day average daily trading volume
    df['20_day_avg_volume'] = df['volume'].rolling(window=20).mean()
    
    # Alpha factor: Difference between current day's volume and 20-day average daily trading volume
    df['volume_diff'] = df['volume'] - df['20_day_avg_volume']
    
    # 20-day cumulative volume delta
    df['volume_delta'] = np.where(df['close'] > df['close'].shift(1), df['volume'], -df['volume'])
    df['20_day_cumulative_volume_delta'] = df['volume_delta'].rolling(window=20).sum()
    
    # 20-day Money Flow Index (MFI)
    df['MFI'] = MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=20)
    
    # 20-day Accumulation/Distribution Line (A/D Line)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['money_flow_multiplier'] = np.where(df['typical_price'] > df['typical_price'].shift(1), 1, 
                                            np.where(df['typical_price'] < df['typical_price'].shift(1), -1, 0))
    df['money_flow_volume'] = df['money_flow_multiplier'] * df['volume']
    df['A/D_Line'] = df['money_flow_volume'].cumsum()
    df['20_day_A/D_Line_change'] = df['A/D_Line'].diff(periods=20)
    
    # 20-day average of the difference between open and close prices
    df['20_day_avg_open_close_diff'] = (df['open'] - df['close']).rolling(window=20).mean()
    
    # 20-day relative strength (RS) of the close price compared to the open price
    df['20_day_RS'] = (df['close'] / df['open']).pct_change(periods=20)
    
    # Select the alpha factors
    alpha_factors = df[['sma_ratio', '20_day_price_return', 'ADX', 'volume_diff', '20_day_cumulative_volume_delta', 
                        'MFI', '20_day_A/D_Line_change', '20_day_avg_open_close_diff', '20_day_RS']]
    
    return alpha_factors
