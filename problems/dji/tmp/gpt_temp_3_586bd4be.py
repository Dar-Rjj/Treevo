import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 5-day and 20-day Simple Moving Averages (SMA)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # Generate a bullish signal when SMA_5 > SMA_20
    df['SMA_Crossover_Signal'] = (df['SMA_5'] > df['SMA_20']).astype(int) - (df['SMA_5'] < df['SMA_20']).astype(int)
    
    # Calculate the True Range (TR)
    df['TR'] = df[['high' - 'low', 
                   abs('high' - df['close'].shift(1)), 
                   abs('low' - df['close'].shift(1))]].max(axis=1)
    
    # Calculate the Positive and Negative Directional Movements (+DM, -DM)
    df['+DM'] = 0
    df['-DM'] = 0
    df.loc[df['high'] - df['high'].shift(1) > df['low'].shift(1) - df['low'], '+DM'] = df['high'] - df['high'].shift(1)
    df.loc[df['low'].shift(1) - df['low'] > df['high'] - df['high'].shift(1), '-DM'] = df['low'].shift(1) - df['low']
    
    # Smooth the +DM and -DM using a 14-period Wilder's Smoothing
    df['+DM_14'] = df['+DM'].ewm(span=14, adjust=False).mean()
    df['-DM_14'] = df['-DM'].ewm(span=14, adjust=False).mean()
    
    # Calculate the Average True Range (ATR)
    df['ATR_14'] = df['TR'].ewm(span=14, adjust=False).mean()
    
    # Calculate the Positive and Negative Directional Indicators (+DI, -DI)
    df['+DI'] = 100 * (df['+DM_14'] / df['ATR_14'])
    df['-DI'] = 100 * (df['-DM_14'] / df['ATR_14'])
    
    # On-Balance Volume (OBV)
    df['OBV'] = 0
    obv_conditions = [
        (df['close'] > df['close'].shift(1), df['volume']),
        (df['close'] < df['close'].shift(1), -df['volume'])
    ]
    df['OBV'] = df.apply(lambda row: sum([cond[1] for cond in obv_conditions if cond[0].loc[row.name]]), axis=1).cumsum()
    
    # Volume-Weighted Average Price (VWAP)
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # Three Line Strike Bearish Pattern
    df['Three_Line_Strike_Bearish'] = 0
    three_line_strike_bearish_conditions = [
        (df['close'] > df['open']),  # Day 1: White candlestick
        (df['close'].shift(1) > df['open'].shift(1)),  # Day 2: White candlestick
        (df['close'].shift(2) > df['open'].shift(2)),  # Day 3: White candlestick
        (df['close'].shift(3) < df['open'].shift(3)),  # Day 4: Large black candlestick
        (df['close'].shift(3) < df['open'].shift(4))  # Day 4: Closes below the opening of Day 1
    ]
    df['Three_Line_Strike_Bearish'] = (df['Three_Line_Strike_Bearish'] == 1).all(axis=1).astype(int)
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (df['SMA_Crossover_Signal'] + 
                          (df['+DI'] - df['-DI']) / 100 + 
                          df['OBV'] + 
                          df['VWAP'] + 
                          df['Three_Line_Strike_Bearish'])
    
    return df['alpha_factor']
