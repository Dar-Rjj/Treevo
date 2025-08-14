import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Simple Moving Averages (SMA)
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_100'] = df['close'].rolling(window=100).mean()

    # Calculate Exponential Moving Averages (EMA)
    df['EMA_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['close'].ewm(span=100, adjust=False).mean()

    # Calculate Rate of Change (RoC)
    df['RoC_5'] = df['close'].pct_change(periods=5)
    df['RoC_10'] = df['close'].pct_change(periods=10)
    df['RoC_20'] = df['close'].pct_change(periods=20)

    # Calculate Relative Strength Index (RSI)
    def rsi(series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI_14'] = rsi(df['close'])

    # Calculate Average True Range (ATR)
    df['TrueRange'] = df[['high', 'low']].apply(lambda x: max(x[0], df['close'].shift(1)) - min(x[1], df['close'].shift(1)), axis=1)
    df['ATR_14'] = df['TrueRange'].rolling(window=14).mean()

    # Calculate Bollinger Bands
    df['SMA_20_BB'] = df['close'].rolling(window=20).mean()
    df['BB_Upper'] = df['SMA_20_BB'] + 2 * df['close'].rolling(window=20).std()
    df['BB_Lower'] = df['SMA_20_BB'] - 2 * df['close'].rolling(window=20).std()

    # Compare daily volume with a moving average of volume
    df['Volume_MA_20'] = df['volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_MA_20']

    # Create a proprietary factor combining SMA and EMA crossovers
    df['SMA_EMA_Crossover'] = (df['SMA_20'] > df['EMA_20']).astype(int) - (df['SMA_20'] < df['EMA_20']).astype(int)

    # Combine all factors into a single alpha factor
    df['Alpha_Factor'] = (
        df['RoC_5'] + 
        df['RoC_10'] + 
        df['RoC_20'] + 
        df['RSI_14'] + 
        df['ATR_14'] + 
        df['Volume_Ratio'] + 
        df['SMA_EMA_Crossover']
    )

    # Return the Alpha Factor
    return df['Alpha_Factor']
