import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['High-Low Spread'] = df['high'] - df['low']
    df['Close-Open Spread'] = df['close'] - df['open']
    df['Intraday Momentum'] = df['High-Low Spread'] - df['Close-Open Spread']

    # Calculate Volume Weighted Average Price (VWAP)
    df['VWAP'] = df['amount'] / df['volume']

    # Determine Volume Synchronization
    df['Logarithmic Volume Change'] = np.log(df['volume'] / df['volume'].shift(1))
    df['Logarithmic Return'] = np.log(df['close'] / df['close'].shift(1))

    # Integrate Price and Volume Dynamics
    df['Integrated Indicator'] = df['Intraday Momentum'] * df['Logarithmic Return']
    df['Combined Indicator'] = df['Integrated Indicator'] + df['VWAP']

    # Enhance Factor with Intraday and Relative Strength
    df['Intraday Trend'] = (df['high'] - df['low']) / df['low']
    df['Rolling Average Close'] = df['close'].rolling(window=50).mean()
    df['Relative Strength'] = df['close'] / df['Rolling Average Close']

    # Incorporate Volatility
    log_returns = np.log(df['close'] / df['close'].shift(1))
    df['Advanced Realized Volatility'] = np.sqrt(log_returns.rolling(window=50).apply(lambda x: (x**2).mean()))

    # Incorporate Liquidity
    df['Average Daily Volume'] = df['volume'].rolling(window=50).mean()
    df['Volume Volatility'] = np.sqrt(((df['volume'] - df['volume'].shift(1))**2).rolling(window=50).mean())

    # Incorporate Market Sentiment
    df['Close-to-Open Ratio'] = df['close'] / df['open'].shift(-1)
    df['Open-to-Close Ratio'] = df['open'] / df['close'].shift(1)

    # Final Alpha Factor
    df['Final Alpha Factor'] = df['Intraday Trend'] * df['Combined Indicator'] * df['Relative Strength'] * df['Advanced Realized Volatility'] * df['Volume Volatility'] * df['Close-to-Open Ratio'] * df['Open-to-Close Ratio']

    return df['Final Alpha Factor']
