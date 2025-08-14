import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute Raw Momentum
    df['Raw_Momentum'] = df['close'].shift(0) - df['close'].shift(20)
    
    # Adjust for Volume
    average_volume_20 = df['volume'].rolling(window=20).mean()
    volume_ratio = df['volume'] / average_volume_20
    df['Volume_Adjusted_Momentum'] = df['Raw_Momentum'] * volume_ratio
    
    # Incorporate True Range Momentum
    df['True_Range'] = df[['high', 'low', 'close']].join(df['close'].shift(1), rsuffix='_prev').apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close_prev']), abs(x['low'] - x['close_prev'])), axis=1
    )
    df['ATR'] = df['True_Range'].ewm(span=14, adjust=False).mean()
    df['True_Range_Momentum'] = df['Raw_Momentum'] / df['ATR']
    
    # Combine Raw and True Range Momentum
    combined_momentum = df['Volume_Adjusted_Momentum'] + df['True_Range_Momentum']
    
    # Volume Confirmation
    vol_ema_5 = df['volume'].ewm(span=5, adjust=False).mean()
    vol_ema_20 = df['volume'].ewm(span=20, adjust=False).mean()
    df['Combined_Momentum'] = combined_momentum * (1.2 if vol_ema_5 > vol_ema_20 else 0.8)
    
    # Incorporate Gaps and Oscillations
    df['Open_to_Close_Gap'] = df['open'] - df['close']
    df['High_Low_Gap'] = df['high'] - df['low']
    average_volume_10 = df['volume'].rolling(window=10).mean()
    df['Volume_Difference'] = df['volume'] - average_volume_10
    df['Volume_Oscillation'] = df['Volume_Difference'] / average_volume_10
    
    # Final Alpha Factor
    df['Alpha_Factor'] = df['Combined_Momentum'] + df['Open_to_Close_Gap'] + df['High_Low_Gap'] + df['Volume_Oscillation']
    
    return df['Alpha_Factor']
