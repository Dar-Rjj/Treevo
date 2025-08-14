import pandas as pd

def heuristics_v2(df):
    def calculate_rsi(series, period=10):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(series, fast=9, slow=21, signal=5):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - emma_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line
    
    def calculate_cmflow(close, high, low, volume, period=15):
        mfv = ((close - low) - (high - close)) / (high - low)
        mfv = mfv.fillna(0.0) * volume
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf
    
    def calculate_mfi(high, low, close, volume, period=10):
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        money_ratio = positive_flow.rolling(window=period).sum() / negative_flow.rolling(window=period).sum()
        return 100 - (100 / (1 + money_ratio))

    rsi = calculate_rsi(df['close'])
    macd_signal = calculate_macd(df['close'])
    cmflow = calculate_cmflow(df['close'], df['high'], df['low'], df['volume'])
    mfi = calculate_mfi(df['high'], df['low'], df['close'], df['volume'])

    heuristics_matrix = (rsi * 0.4 + macd_signal * 0.2 + cmflow * 0.3 + mfi * 0.1)
    return heuristics_matrix
