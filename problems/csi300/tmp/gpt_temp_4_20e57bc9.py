import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Raw Momentum
    df['Close_t-20'] = df['close'].shift(20)
    df['RawMomentum'] = df['close'] - df['Close_t-20']
    
    # Adjust for Volume
    df['AvgVol'] = df['volume'].rolling(window=20).mean()
    df['VolRatio'] = df['volume'] / df['AvgVol']
    df['AdjMomentum'] = df['RawMomentum'] * df['VolRatio']
    
    # Incorporate Enhanced Price Gaps
    df['GapOC'] = df['open'] - df['close']
    df['GapHL'] = df['high'] - df['low']
    df['CombinedMomentum'] = df['AdjMomentum'] + df['GapOC'] + df['GapHL']
    
    # Confirm with Volume Trend
    df['VolEma5'] = df['volume'].ewm(span=5, adjust=False).mean()
    df['VolMA20'] = df['volume'].rolling(window=20).mean()
    df['ConfirmedMomentum'] = df['CombinedMomentum'].where(df['VolEma5'] > df['VolMA20'], df['CombinedMomentum'] * 0.8) * (1.2 if df['VolEma5'] > df['VolMA20'] else 1.0)
    
    # Adjust by ATR
    df['High-Low'] = df['high'] - df['low']
    df['High-Close_t-1'] = (df['high'] - df['close'].shift(1)).abs()
    df['Low-Close_t-1'] = (df['low'] - df['close'].shift(1)).abs()
    df['TrueRange'] = df[['High-Low', 'High-Close_t-1', 'Low-Close_t-1']].max(axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    df['FinalFactor'] = df['ConfirmedMomentum'] / df['ATR']
    
    # Enhanced Price Reversal Sensitivity
    df['High-Close_Spread'] = df['high'] - df['close']
    df['Open-Low_Spread'] = df['open'] - df['low']
    df['Weighted_High-Close_Spread'] = df['High-Close_Spread'] * df['volume']
    df['Weighted_Open-Low_Spread'] = df['Open-Low_Spread'] * df['volume']
    df['Combined_Weighted_Spreads'] = df['Weighted_High-Close_Spread'] + df['Weighted_Open-Low_Spread']
    
    # Calculate Daily Gaps and Volume Weighted Average of Gaps
    df['Daily_Gap'] = df['open'] - df['close'].shift(1)
    df['Volume_Weighted_Daily_Gap'] = (df['Daily_Gap'] * df['volume']).sum() / df['volume'].sum()
    
    # Incorporate Price Gaps
    df['Open-to-Close_Gap'] = df['open'] - df['close']
    df['Combined_Volume_Adjusted_Factors'] = df['FinalFactor'] + df['Open-to-Close_Gap'] + df['Volume_Weighted_Daily_Gap']
    
    # Calculate Daily Range and Smoothing Factor for High-Low Spread
    df['Daily_Range'] = df['high'] - df['low']
    df['Smoothed_High-Low_Spread_EMA'] = df['Daily_Range'].ewm(span=5, adjust=False).mean()
    
    # Final Factor
    df['Final_Factor'] = (df['Combined_Volume_Adjusted_Factors'] + df['high'] - df['close']) - df['Combined_Weighted_Spreads'] + df['Open-to-Close_Gap'] + df['Volume_Weighted_Daily_Gap'] - df['Smoothed_High-Low_Spread_EMA']
    
    return df['Final_Factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# factor_values = heuristics_v2(df)
# print(factor_values)
