importance of the logarithm of the closing price to ATR ratio.}

```python
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Chaikin Money Flow (CMF) with a 10-day window
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['volume']
    cmf = money_flow_volume.rolling(window=10).sum() / df['volume'].rolling(window=10).sum()
    
    # Average True Range (ATR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    
    # Logarithm of Closing Price to ATR Ratio
    close_to_atr_ratio_log = np.log(df['close'] / atr)
    
    # 10-day Exponential Moving Average (EMA) of the closing price
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    
    # Composite heuristic matrix with adjusted weights
    heuristics_matrix = 0.2 * cmf + 0.6 * close_to_atr_ratio_log + 0.2 * (ema_10 / df['close'])
    
    return heuristics_matrix
