import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily return
    df['Daily_Return'] = df['close'].pct_change()

    # Calculate intraday momentum
    df['Intraday_Momentum'] = (df['close'] - df['open']) / df['open']

    # Identify trend strength
    df['MA_50'] = df['close'].rolling(window=50).mean()
    df['MA_200'] = df['close'].rolling(window=200).mean()

    # Analyze volume and amount
    df['Volume_Change'] = df['volume'].pct_change()
    df['Amount_Change'] = df['amount'].pct_change()

    # Volume-weighted price
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    # Combine price and volume information
    df['Price_Volume_Divergence'] = (df['Daily_Return'] - df['Volume_Change']) / df['Daily_Return']

    # Consider volatility
    df['True_Range'] = df[['high', 'low', 'close']].diff(axis=1).abs().max(axis=1)
    df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()

    # Incorporate pattern recognition
    df['Doji'] = (np.abs(df['open'] - df['close']) < 0.005 * (df['high'] - df['low'])).astype(int)
    df['Hammer_Inverted_Hammer'] = (
        ((df['close'] - df['low']) / (df['high'] - df['low'])) > 0.6 &
        ((df['high'] - df['close']) / (df['high'] - df['low'])) < 0.1
    ).astype(int)

    # Examine market sentiment
    df['High_Low_Ratio'] = (df['high'] - df['low']) / df['close']
    df['Open_Close_Ratio'] = (df['close'] - df['open']) / df['open']

    # Analyze trading activity
    df['Turnover_Ratio'] = df['volume'] / 1e6  # Assuming 1 million shares as outstanding
    df['Activity_Index'] = (df['volume'] + df['amount']) / 2

    # Explore additional indicators
    def rsi(series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    df['RSI'] = rsi(df['close'])
    df['MACD_Line'] = ema(df['close'], 12) - ema(df['close'], 26)
    df['Signal_Line'] = ema(df['MACD_Line'], 9)
    df['MACD_Histogram'] = df['MACD_Line'] - df['Signal_Line']

    # Combine all factors into a single alpha factor
    alpha_factor = (
        df['Daily_Return'] +
        df['Intraday_Momentum'] +
        (df['MA_50'] - df['MA_200']) / df['MA_200'] +
        df['Volume_Change'] +
        df['Amount_Change'] +
        df['VWAP'] +
        df['Price_Volume_Divergence'] +
        df['True_Range'] +
        df['Volatility_20'] +
        df['Doji'] +
        df['Hammer_Inverted_Hammer'] +
        df['High_Low_Ratio'] +
        df['Open_Close_Ratio'] +
        df['Turnover_Ratio'] +
        df['Activity_Index'] +
        df['RSI'] / 100 +
        df['MACD_Histogram']
    )

    return alpha_factor
