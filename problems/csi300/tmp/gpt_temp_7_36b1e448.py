import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Trend Indicators
    df['5_day_MA'] = df['close'].rolling(window=5).mean()
    df['20_day_MA'] = df['close'].rolling(window=20).mean()
    df['trend_signal'] = df['5_day_MA'] - df['20_day_MA']
    df['price_change_20d'] = df['close'].pct_change(20)

    # Momentum Indicators
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI'] = rsi(df['close'])
    df['ROC_10'] = df['close'].pct_change(10)
    df['ROC_20'] = df['close'].pct_change(20)

    # Volume Indicators
    obv = [0]
    for i in range(1, len(df)):
        if df['close'][i] > df['close'][i-1]:
            obv.append(obv[-1] + df['volume'][i])
        elif df['close'][i] < df['close'][i-1]:
            obv.append(obv[-1] - df['volume'][i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    df['volume_change'] = df['volume'].pct_change()

    # Multi-Day Patterns
    df['bullish_engulfing'] = ((df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))).astype(int)
    df['bearish_engulfing'] = ((df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))).astype(int)
    avg_volume = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = (df['volume'] > 1.5 * avg_volume).astype(int)

    # Volatility Indicators
    df['20_day_std'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['20_day_MA'] + 2 * df['20_day_std']
    df['lower_band'] = df['20_day_MA'] - 2 * df['20_day_std']
    df['ATR'] = df['high'].combine(df['low'], max) - df['low'].combine(df['high'], min)
    df['ATR'] = df['ATR'].ewm(span=14, adjust=False).mean()

    # Market Sentiment Indicators
    # Assuming we have a sentiment score column and put-call ratio column
    # df['sentiment_score'] = ...  # Calculate or obtain sentiment scores
    # df['put_call_ratio'] = ...  # Calculate or obtain put-call ratios

    # Combine all factors into a single alpha factor
    alpha_factor = (df['trend_signal'] + df['price_change_20d'] + 
                    df['RSI'] + df['ROC_10'] + df['ROC_20'] + 
                    df['OBV'] + df['volume_change'] + 
                    df['bullish_engulfing'] - df['bearish_engulfing'] + 
                    df['volume_spike'] + 
                    (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band']) + 
                    df['ATR'])
    
    return alpha_factor
