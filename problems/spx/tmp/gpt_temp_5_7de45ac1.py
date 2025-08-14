import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # 1.1.1. Simple Moving Average (SMA) Crossover
    df['SMA5'] = df['close'].rolling(window=5).mean()
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA_Crossover'] = (df['SMA5'] > df['SMA20']).astype(int)

    # 1.1.2. Rate of Change (ROC)
    df['ROC10'] = df['close'].pct_change(periods=10)
    roc_threshold = 0.05
    df['Overbought'] = (df['ROC10'] > roc_threshold).astype(int)
    df['Oversold'] = (df['ROC10'] < -roc_threshold).astype(int)

    # 1.2.1. True Range
    df['True_Range'] = df[['high', 'low']].apply(lambda x: max(x['high'], df['close'].shift(1)) - min(x['low'], df['close'].shift(1)), axis=1)

    # 1.2.2. Average True Range (ATR)
    df['ATR14'] = df['True_Range'].rolling(window=14).mean()

    # 2.1.1. Bullish Engulfing
    df['Bullish_Engulfing'] = ((df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1)) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))).astype(int)

    # 2.1.2. Bearish Engulfing
    df['Bearish_Engulfing'] = ((df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1)) & (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))).astype(int)

    # 3.1. On-Balance Volume (OBV)
    df['OBV'] = (df['close'] > df['close'].shift(1)).astype(int) * df['volume'] - (df['close'] < df['close'].shift(1)).astype(int) * df['volume']
    df['OBV'] = df['OBV'].cumsum()

    # 3.2. Volume Spike
    avg_volume = df['volume'].rolling(window=20).mean()
    df['Volume_Spike'] = (df['volume'] > 2 * avg_volume).astype(int)

    # 4.1. Price and Volume Convergence
    df['SMA_OBV_Confirmation'] = (df['SMA_Crossover'] == 1) & (df['OBV'] > df['OBV'].shift(1))
    df['SMA_OBV_Confirmation'] = df['SMA_OBV_Confirmation'].astype(int)

    # 4.2. Pattern Confirmation
    df['Bullish_Engulfing_Confirmed'] = (df['Bullish_Engulfing'] == 1) & (df['ATR14'] < df['True_Range'])
    df['Bullish_Engulfing_Confirmed'] = df['Bullish_Engulfing_Confirmed'].astype(int)

    # Final Alpha Factor
    df['Alpha_Factor'] = (
        df['SMA_Crossover'] + 
        df['Overbought'] + 
        df['Oversold'] + 
        df['Bullish_Engulfing_Confirmed'] - 
        df['Bearish_Engulfing'] + 
        df['SMA_OBV_Confirmation'] + 
        df['Volume_Spike']
    )

    return df['Alpha_Factor']
