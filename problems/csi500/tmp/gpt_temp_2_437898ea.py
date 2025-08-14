import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    df['True_Range'] = df[['high', 'low']].apply(lambda x: x['high'] - x['low'], axis=1)
    
    # Calculate Average True Range (ATR) over a period of 14 days
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()
    
    # Calculate short-term (5 days) and long-term (20 days) moving averages
    df['MA_5'] = df['close'].rolling(window=5).mean()
    df['MA_20'] = df['close'].rolling(window=20).mean()
    
    # Create a moving average crossover signal
    df['MA_Crossover'] = np.where(df['MA_5'] > df['MA_20'], 1, 0)
    
    # Calculate the rate of change (ROC) of closing prices for different periods
    df['ROC_10'] = df['close'].pct_change(periods=10)
    df['ROC_20'] = df['close'].pct_change(periods=20)
    df['ROC_60'] = df['close'].pct_change(periods=60)
    
    # Calculate Chaikin Money Flow (CMF) without normalization over 20 days
    df['Money_Flow_Volume'] = (df['close'] - df['low']) - (df['high'] - df['close']) * df['volume'] / (df['high'] - df['low'])
    df['CMF_20'] = df['Money_Flow_Volume'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Compute Bollinger Bands
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['Bollinger_Upper'] = rolling_mean + 2 * rolling_std
    df['Bollinger_Lower'] = rolling_mean - 2 * rolling_std
    df['BB_Width'] = df['Bollinger_Upper'] - df['Bollinger_Lower']
    
    # Calculate Keltner Channels using ATR
    df['Keltner_Upper'] = df['MA_20'] + 2 * df['ATR_14']
    df['Keltner_Lower'] = df['MA_20'] - 2 * df['ATR_14']
    df['Keltner_Width'] = df['Keltner_Upper'] - df['Keltner_Lower']
    
    # Detect and quantify common candlestick patterns
    def detect_candlestick_patterns(open, high, low, close):
        bullish_engulfing = (open.shift(1) > close.shift(1)) & (close > open) & (close > open.shift(1)) & (open < close.shift(1))
        bearish_engulfing = (open.shift(1) < close.shift(1)) & (close < open) & (close < open.shift(1)) & (open > close.shift(1))
        doji = (high - low) / (high + low) < 0.005
        hammer = (close > open) & ((close - open) / (high - low) > 0.6) & ((open - low) / (high - low) > 0.6)
        return pd.DataFrame({
            'Bullish_Engulfing': bullish_engulfing,
            'Bearish_Engulfing': bearish_engulfing,
            'Doji': doji,
            'Hammer': hammer
        })
    
    candlestick_patterns = detect_candlestick_patterns(df['open'], df['high'], df['low'], df['close'])
    
    # Combine all factors into a single DataFrame
    factors = pd.concat([df[['ATR_14', 'MA_Crossover', 'ROC_10', 'ROC_20', 'ROC_60', 'CMF_20', 'BB_Width', 'Keltner_Width']], candlestick_patterns], axis=1)
    
    return factors
