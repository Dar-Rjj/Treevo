import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Momentum Indicator
    df['ROC_12'] = (df['close'].pct_change(periods=12) * 100)
    df['SMA_ROC_25'] = df['ROC_12'].rolling(window=25).mean()
    df['Momentum_Signal'] = df['SMA_ROC_25'].apply(lambda x: 1 if x > 0 else -1)

    # Volume-based Strength Indicator
    df['VMA_20'] = df['volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['volume'] / df['VMA_20']
    df['Volume_Score'] = df['Volume_Ratio'].apply(lambda x: 1 if x > 1 else (0 if x == 1 else -1))

    # Price Volatility Indicator
    df['Daily_Range'] = df['high'] - df['low']
    df['ATR_14'] = df['Daily_Range'].rolling(window=14).mean()
    df['ATR_Trend'] = df['ATR_14'].diff().apply(lambda x: 1 if x > 0 else -1)

    # Money Flow Index (MFI)
    df['Typical_Price'] = (df['high'] + df['low'] + df['close']) / 3
    df['Money_Flow'] = df['Typical_Price'] * df['volume']
    positive_flow = df[df['Typical_Price'] > df['Typical_Price'].shift(1)]['Money_Flow']
    negative_flow = df[df['Typical_Price'] < df['Typical_Price'].shift(1)]['Money_Flow']
    df['Positive_Money_Flow_14'] = positive_flow.rolling(window=14).sum()
    df['Negative_Money_Flow_14'] = negative_flow.rolling(window=14).sum().abs()
    df['Money_Flow_Ratio'] = df['Positive_Money_Flow_14'] / df['Negative_Money_Flow_14']
    df['MFI'] = 100 - (100 / (1 + df['Money_Flow_Ratio']))
    conditions = [
        (df['MFI'] < 20),
        (df['MFI'] > 80),
        (df['MFI'] >= 20) & (df['MFI'] <= 80)
    ]
    choices = ['oversold', 'overbought', 'neutral']
    df['MFI_Condition'] = pd.np.select(conditions, choices, default='neutral')

    # Accumulation/Distribution Line (A/D Line)
    df['Money_Flow_Multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    df['A_D_Line'] = (df['Money_Flow_Multiplier'] * df['volume']).cumsum()
    df['A_D_Trend'] = df['A_D_Line'].diff().apply(lambda x: 1 if x > 0 else -1)

    # Combine all indicators into a single alpha factor
    df['Alpha_Factor'] = df['Momentum_Signal'] + df['Volume_Score'] + df['ATR_Trend'] + df['A_D_Trend']

    return df['Alpha_Factor'].dropna()

# Example usage:
# df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'], index=pd.date_range(start="2023-01-01", periods=len(data), freq='D'))
# alpha_factor = heuristics_v2(df)
