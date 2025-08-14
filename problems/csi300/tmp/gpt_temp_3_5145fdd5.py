import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate simple moving averages (SMA)
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()

    # Calculate exponential moving averages (EMA)
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # Trend strength indicator: Difference between 20-day EMA and 50-day EMA
    df['Trend_Strength'] = df['EMA_20'] - df['EMA_50']

    # Rate of change (ROC) of the 50-day EMA
    df['ROC_50_EMA'] = df['EMA_50'].pct_change(periods=1)

    # Relative Strength Index (RSI) over the past 14 days
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Volume spike flag
    df['Volume_20_MA'] = df['volume'].rolling(window=20).mean()
    df['Volume_Spike_Flag'] = (df['volume'] > df['Volume_20_MA'] * 1.5).astype(int)

    # Correlation between daily volume and price changes over the past 30 days
    df['Price_Change'] = df['close'].pct_change()
    df['Vol_Price_Correlation'] = df['Price_Change'].rolling(window=30).corr(df['volume'])

    # Volume-Weighted Average Price (VWAP)
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['Close_VWAP_Ratio'] = df['close'] / df['VWAP']

    # Daily price range and its ratio to the closing price
    df['Daily_Range'] = df['high'] - df['low']
    df['Range_Close_Ratio'] = df['Daily_Range'] / df['close']

    # Opening price gap
    df['Prev_Close'] = df['close'].shift(1)
    df['Opening_Gap'] = df['open'] - df['Prev_Close']

    # True range
    df['True_Range'] = df[['high', 'low', 'Prev_Close']].apply(
        lambda x: max(x[0] - x[1], x[0] - x[2], x[2] - x[1]), axis=1)

    # Average true range (ATR)
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()
    df['ATR_Close_Ratio'] = df['ATR_14'] / df['close']

    # Composite factor
    df['Composite_Factor'] = (
        0.3 * df['Trend_Strength'] +
        0.2 * df['ROC_50_EMA'] +
        0.1 * df['RSI'] +
        0.1 * df['MACD'] +
        0.1 * df['Volume_Spike_Flag'] +
        0.1 * df['Vol_Price_Correlation'] +
        0.1 * df['Close_VWAP_Ratio']
    )

    return df['Composite_Factor']
