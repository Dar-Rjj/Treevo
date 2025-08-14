import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Adaptive window calculation for EMA, RSI, and VWAP
    ema_window = 7 + (df['close'].rolling(window=30).std() / df['close'].rolling(window=120).std()).apply(lambda x: int(7 * x))
    rsi_window = 14 + (df['close'].rolling(window=30).std() / df['close'].rolling(window=120).std()).apply(lambda x: int(14 * x))
    vwap_window = 7 + (df['volume'].rolling(window=30).std() / df['volume'].rolling(window=120).std()).apply(lambda x: int(7 * x))

    # Volume-Weighted Average Price (VWAP)
    vwap = (df['amount'] / df['volume']).rolling(window=vwap_window, min_periods=1).mean()
    
    # Exponential Moving Average (EMA)
    ema_close = df['close'].ewm(span=ema_window, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Logarithmic Returns
    log_returns = np.log(df['close'] / df['close'].shift(5))
    
    # Sentiment Factor
    sentiment = (df['close'] - df['open']) / df['open']
    
    # Dynamic Volatility
    volatility = df['close'].rolling(window=30).std()

    # Alpha Factor
    factor = ((df['close'] - ema_close) / volatility) * rsi * log_returns * (1 + sentiment)

    return factor
