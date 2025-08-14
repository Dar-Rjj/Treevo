import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Spread
    df['High_Low_Spread'] = df['high'] - df['low']

    # Calculate 20-day Exponential Moving Average of High-Low Spreads
    df['HL_Spread_EMA_20'] = df['High_Low_Spread'].ewm(span=20, adjust=False).mean()

    # Calculate High-Low Spread Oscillator
    df['HL_Spread_Oscillator'] = (df['High_Low_Spread'] - df['HL_Spread_EMA_20']) / df['HL_Spread_EMA_20']

    # Calculate Volume Ratio
    df['Volume_EMA_20'] = df['volume'].ewm(span=20, adjust=False).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_EMA_20']

    # Adjust High-Low Spread Oscillator
    df['Adjusted_HL_Oscillator'] = df['HL_Spread_Oscillator'] * df['Volume_Ratio']
    df['Adjusted_HL_Oscillator'] = df['Adjusted_HL_Oscillator'].clip(lower=-3, upper=3)

    # Calculate Short-Term Momentum
    df['Close_EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['Short_Term_Momentum'] = df['close'] - df['Close_EMA_10']

    # Calculate Long-Term Momentum
    df['Close_EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['Long_Term_Momentum'] = df['close'] - df['Close_EMA_20']

    # Determine Relative Strength
    df['Relative_Strength_Score'] = (df['Short_Term_Momentum'] > df['Long_Term_Momentum']).astype(int)

    # Compute Short-Term and Long-Term Volume Trends
    df['Volume_Trend_10'] = df['volume'].ewm(span=10, adjust=False).sum()
    df['Volume_Trend_20'] = df['volume'].ewm(span=20, adjust=False).sum()
    df['Volume_Ratio_Score'] = df['Volume_Trend_10'] / df['Volume_Trend_20']

    # Combine Scores
    df['Final_Dynamic_Score'] = 1 - (df['Relative_Strength_Score'] * df['Volume_Ratio_Score'])

    # Apply Dynamic Score to Adjusted Close Price
    df['Dynamic_Adjusted_Close'] = df['close'] * df['Final_Dynamic_Score']

    # Calculate High-Low Difference and Subtract Open Price
    df['High_Low_Difference'] = df['high'] - df['low']
    df['Open_Price_Difference'] = df['High_Low_Difference'] - df['open']
    df['Absolute_Movement'] = df['Open_Price_Difference'].abs()

    # Smooth Over N Days
    df['Smoothed_Absolute_Movement'] = df['Absolute_Movement'].ewm(span=20, adjust=False).mean()

    # Compute Intraday Volatility
    df['Intraday_Volatility'] = (df['high'] - df['low']) / df['close']

    # Calculate Price Momentum
    df['10_Day_Return'] = df['close'].pct_change(10)
    df['Adjusted_Momentum'] = df['10_Day_Return'] * df['Intraday_Volatility']

    # Combine smoothed momentum with open price trend
    df['Combined_Factor'] = df['Adjusted_Momentum'] + df['Smoothed_Absolute_Movement']
    df['Open_Price_Trend'] = df['open'].diff().ewm(span=10, adjust=False).mean()
    df['Final_Combined_Factor'] = df['Combined_Factor'] * df['Open_Price_Trend']

    # Final Alpha Factor
    df['Final_Alpha_Factor'] = df['Adjusted_HL_Oscillator'] + df['Final_Combined_Factor'] + df['Dynamic_Adjusted_Close']

    return df['Final_Alpha_Factor']
