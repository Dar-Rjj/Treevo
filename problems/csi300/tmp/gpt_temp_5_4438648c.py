import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Intraday Return
    df['Volume-Weighted Intraday Return'] = (df['high'] - df['low']) / df['open'] * df['volume']
    
    # Calculate Intraday Momentum
    df['High-Low Range'] = df['high'] - df['low']
    df['Intraday Percentage Change'] = df['High-Low Range'] / df['open']

    # Calculate VWMA
    df['Short-Term VWMA'] = (df['close'] * df['volume']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    df['Long-Term VWMA'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    df['VWMA Momentum'] = df['Short-Term VWMA'] - df['Long-Term VWMA']

    # Combine Intraday Momentum and Volume-Weighted Intraday Return
    df['Momentum-Return Combined'] = df['Intraday Percentage Change'] * df['Volume-Weighted Intraday Return']

    # Calculate Adjusted Reversal Indicator
    df['Next Day Volume-Weighted Intraday Return'] = df['Volume-Weighted Intraday Return'].shift(-1)
    df['Adjusted Reversal'] = df['Next Day Volume-Weighted Intraday Return'] - df['Volume-Weighted Intraday Return']

    # Measure Volume Stability
    df['Volume Change'] = df['volume'] - df['volume'].shift(1)
    df['Absolute Volume Change'] = df['Volume Change'].abs()
    df['Volume Stability'] = df['Absolute Volume Change'].rolling(window=5).sum()

    # Calculate Volume Flow
    df['Volume Difference'] = df['volume'] - df['volume'].shift(1)
    df['Average Volume'] = (df['volume'] + df['volume'].shift(1)) / 2
    df['Volume Flow'] = df['Volume Difference'] / df['Average Volume']

    # Incorporate Trading Volume and Amount
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['Amount-Weighted Price'] = (df['close'] * df['amount']).cumsum() / df['amount'].cumsum()

    # Combine VWMA Momentum, Intraday Momentum, and Volume Flow
    df['Combined Factor'] = df['VWMA Momentum'] * df['Intraday Percentage Change'] * df['Volume Flow']

    # Weight by Intraday Volatility
    df['True Range'] = df[['high' - 'low', ('high' - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()]].max(axis=1)
    df['ATR_5'] = df['True Range'].rolling(window=5).mean()
    df['Combined Factor Weighted'] = df['Combined Factor'] * df['ATR_5']

    # Final Alpha Factor
    df['Final Alpha Factor'] = df['Momentum-Return Combined'] * df['Adjusted Reversal'] * df['Combined Factor Weighted']

    # Determine Momentum Signal
    df['Momentum Signal'] = df['Final Alpha Factor'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # Assign Weight by Volume
    df['Latest Volume'] = df['volume'].shift(1)
    df['Weighted Factor'] = df['Final Alpha Factor'] * df['Latest Volume']

    # Summarize Over Multiple Days
    df['Cumulative Weighted Factor 5D'] = df['Weighted Factor'].rolling(window=5).sum()
    df['Cumulative Weighted Return 10D'] = df['Final Alpha Factor'].rolling(window=10).sum()

    return df['Cumulative Weighted Factor 5D']
