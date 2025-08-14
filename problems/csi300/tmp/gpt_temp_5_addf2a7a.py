import numpy as np
def heuristics_v2(df):
    # Calculate Short-Term and Long-Term Volume-Weighted Moving Averages (VWMA)
    df['Short_Term_VWMA'] = (df['close'] * df['volume']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    df['Long_Term_VWMA'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

    # Subtract Short-Term from Long-Term VWMA
    df['VWMA_Diff'] = df['Short_Term_VWMA'] - df['Long_Term_VWMA']

    # Determine Adaptive Crossover Signal
    df['Crossover_Signal'] = np.where(df['VWMA_Diff'] > 0, 1, -1)
    df['Crossover_Signal'] *= df['volume'].ewm(span=5, adjust=False).mean()

    # Calculate Intraday Momentum
    df['Intraday_Range'] = df['high'] - df['low']
    df['Intraday_Momentum'] = df['Intraday_Range'] / df['open']

    # Calculate Volume-Weighted Intraday Return
    df['Volume_Weighted_Intraday_Return'] = df['Intraday_Range'] * df['volume']

    # Combine Intraday Momentum and Volume-Weighted Intraday Return
    avg_volume = (df['volume'].shift(1) + df['volume']) / 2
    df['Combined_Intraday_Factor'] = df['Intraday_Momentum'] * df['Volume_Weighted_Intraday_Return'] * avg_volume

    # Calculate Volume Flow
    volume_diff = df['volume'] - df['volume'].shift(1)
    avg_vol = (df['volume'].shift(1) + df['volume']) / 2
    df['Volume_Flow'] = volume_diff / avg_vol

    # Weight Combined Intraday Factor by Volume Flow
    df['Weighted_Intraday_Factor'] = df['Combined_Intraday_Factor'] * df['Volume_Flow']

    # Compute Intraday Volatility
    prices = df[['open', 'high', 'low', 'close']]
    intraday_volatility = prices.rolling(window=5).std().mean(axis=1)
    intraday_volatility_smoothed = intraday_volatility.ewm(span=5, adjust=False).mean()

    # Adjust Final Weighted Intraday Factor
    df['Final_Intraday_Alpha'] = df['Weighted_Intraday_Factor'] * intraday_volatility_smoothed

    # Integrate Crossover and Final Intraday Factors
    df['Integrated_Alpha_Factor'] = df['Crossover_Signal'] + df['Final_Intraday_Alpha']

    # Calculate Reversal Indicator
    df['Volume_Weighted_Intraday_Return_Shifted'] = df['Volume_Weighted_Intraday_Return'].shift(1)
    df['Reversal_Indicator'] = (df['Volume_Weighted_Intraday_Return'] - df['Volume_Weighted_Intraday_Return_Shifted']) / (df['volume'] + df['volume'].shift(1))

    # Combine Integrated Alpha Factor and Reversal Indicator
    df['Enhanced_Alpha'] = df['Integrated_Alpha_Factor'] * df['Reversal_Indicator']

    # Integrate Price Trend
    df['Price_Trend'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)

    # Add a New Component: Intraday Price Range to Volume Ratio
    df['Intraday_Price_Range'] = df['high'] - df['low']
    df['Volume_Ratio'] = df['volume'] / df['volume'].shift(1)
    df['Intraday_Price_Range_to_Volume_Ratio'] = df['Intraday_Price_Range'] * df['Volume_Ratio']

    # Introduce Volume-Weighted Intraday Volatility
    df['Squared_High_Low_Range'] = (df['high'] - df['low']) ** 2
    df['Volume_Weighted_Intraday_Volatility'] = df['Squared_High_Low_Range'] * df['volume']

    # Final Combined Alpha Factor
    df['Final_Combined_Alpha_Factor'] = df['Enhanced_Alpha'] * df['Intraday_Price_Range_to_Volume_Ratio'] * df['Volume_Weighted_Intraday_Volatility']

    return df['
