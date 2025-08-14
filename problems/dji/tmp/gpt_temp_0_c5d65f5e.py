import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Volatility
    df['True Range'] = df[['High', 'Low', 'Close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['Close'].shift(1)), abs(x[1] - df['Close'].shift(1))), axis=1)
    df['Intraday Volatility'] = df['True Range'].rolling(window=20).mean()

    # Calculate Momentum
    df['Momentum'] = df['Close'] - df['Close'].rolling(window=20).mean()
    
    # Incorporate Volume Impact
    df['Normalized Volume'] = (df['Volume'] - df['Volume'].rolling(window=20).min()) / (df['Volume'].rolling(window=20).max() - df['Volume'].rolling(window=20).min())
    df['Volume Weighted Momentum'] = df['Momentum'] * df['Normalized Volume']

    # Adjust Momentum by Inverse of Volatility and add Volume Weighted Momentum
    df['Adjusted Momentum'] = df['Momentum'] / df['Intraday Volatility'] + df['Volume Weighted Momentum']

    # Calculate Relative Strength
    df['Relative Strength'] = df['Close'] / df['Close'].shift(20)

    # Measure Volume Activity Change
    df['Average Volume'] = df['Volume'].rolling(window=20).mean()
    df['Volume Change'] = df['Volume'] - df['Average Volume']

    # Combine Relative Strength and Volume Change
    df['Intermediate Factor'] = df['Relative Strength'] * df['Volume Change']

    # Calculate Daily Price Momentum
    df['Daily Price Momentum'] = df['Close'] - df['Close'].shift(1)

    # Calculate Short-Term Trend
    df['Short-Term Trend'] = df['Daily Price Momentum'].rolling(window=5).mean()

    # Generate Integrated Alpha Factor
    df['Integrated Alpha Factor'] = (df['Momentum'] - df['Short-Term Trend']) * df['Intermediate Factor']

    return df['Integrated Alpha Factor']
