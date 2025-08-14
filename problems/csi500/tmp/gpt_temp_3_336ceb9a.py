import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Movement Range
    df['Price_Range'] = df['high'] - df['low']
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Determine Daily Return Deviation from VWAP
    df['Return_Deviation'] = df['close'] - df['VWAP']
    
    # Identify Trend Reversal Potential
    df['Trend_Reversal_Potential'] = (df['Return_Deviation'] > df['Return_Deviation'].shift(1)).astype(int)
    
    # Check for Volume Increase
    df['Volume_MA_5'] = df['volume'].rolling(window=5).mean()
    df['Volume_Increase'] = (df['volume'] > df['Volume_MA_5']).astype(int)
    
    # Calculate Intraday Return
    df['Intraday_Return'] = (df['high'] - df['low']) / df['open']
    
    # Adjust for Volume
    df['Volume_MA_7'] = df['volume'].rolling(window=7).mean()
    df['Volume_Adjustment'] = (df['volume'] - df['Volume_MA_7'])
    df['Adjusted_Intraday_Return'] = df['Intraday_Return'] * df['Volume_Adjustment']
    
    # Incorporate Price Volatility
    df['Close_STD_7'] = df['close'].rolling(window=7).std()
    df['Intraday_Return_Adjusted_Volatility'] = df['Intraday_Return'] * (
        1.4 if df['Close_STD_7'] > df['Close_STD_7'].median() else 0.6
    )
    
    # Incorporate Momentum Shift
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['Momentum_Shift'] = 0
    df.loc[df['SMA_5'] > df['SMA_20'], 'Momentum_Shift'] = 0.5
    df.loc[df['SMA_5'] < df['SMA_20'], 'Momentum_Shift'] = -0.5
    
    # Combine Indicators
    df['Combined_Indicator'] = (
        df['Trend_Reversal_Potential'] * 0.3 +
        df['Volume_Increase'] * 0.2 +
        df['Adjusted_Intraday_Return'] * 0.4 +
        df['Momentum_Shift'] * 0.1
    )
    
    # Smoothing the combined score
    df['Smoothed_Combined_Indicator'] = df['Combined_Indicator'].rolling(window=3).mean()
    
    # Use as a Factor for Stock Selection or Timing
    return df['Smoothed_Combined_Indicator']
