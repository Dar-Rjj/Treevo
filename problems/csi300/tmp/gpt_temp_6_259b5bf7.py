import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate VWAP
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    total_volume = df['Volume'].sum()
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    # VWAP Difference from Open and Close
    df['VWAP_Diff_Open'] = df['VWAP'] - df['Open']
    df['VWAP_Diff_Close'] = df['VWAP'] - df['Close']

    # Calculate Short-Term Momentum
    df['10_day_Price_Change'] = df['Close'] - df['Close'].shift(10)

    # Calculate Long-Term Momentum
    df['60_day_Price_Change'] = df['Close'] - df['Close'].shift(60)

    # Calculate Short-Term Volatility
    df['Daily_Returns'] = df['Close'].diff()
    df['10_day_Volatility'] = df['Daily_Returns'].rolling(window=10).std()

    # Calculate Long-Term Volatility
    df['60_day_Volatility'] = df['Daily_Returns'].rolling(window=60).std()

    # Measure Liquidity
    df['Average_Volume_20_days'] = df['Volume'].rolling(window=20).mean()
    df['Liquidity_Indicator'] = df['Volume'] / df['Average_Volume_20_days']

    # Combine Momentum, Volatility, and Liquidity
    df['Short_Term_Momentum_and_Volatility_Product'] = df['10_day_Price_Change'] * df['10_day_Volatility']
    df['Long_Term_Momentum_and_Volatility_Product'] = df['60_day_Price_Change'] * df['60_day_Volatility']
    
    df['Adjusted_Short_Term_Product'] = df['Short_Term_Momentum_and_Volatility_Product'] * df['Liquidity_Indicator']
    df['Adjusted_Long_Term_Product'] = df['Long_Term_Momentum_and_Volatility_Product'] * df['Liquidity_Indicator']

    # VWAP-Momentum-Liquidity Composite
    df['VWAP_Adjusted_Short_Term_Momentum'] = df['VWAP_Diff_Open'] * df['Adjusted_Short_Term_Product']
    df['VWAP_Adjusted_Long_Term_Momentum'] = df['VWAP_Diff_Close'] * df['Adjusted_Long_Term_Product']

    # Introduce Open-Close Price Spread
    df['Open_Close_Spread'] = df['Close'] - df['Open']
    df['Adjusted_Open_Close_Spread'] = df['Open_Close_Spread'] * df['Liquidity_Indicator']

    # Final Alpha Factor
    df['Final_Alpha_Factor'] = (
        df['VWAP_Adjusted_Short_Term_Momentum'] -
        df['VWAP_Adjusted_Long_Term_Momentum'] +
        df['Adjusted_Short_Term_Product'] -
        df['Adjusted_Long_Term_Product'] +
        df['Adjusted_Open_Close_Spread']
    )

    return df['Final_Alpha_Factor']
