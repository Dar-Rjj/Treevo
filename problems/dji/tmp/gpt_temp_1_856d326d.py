import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Momentum
    high_low_diff = df['high'] - df['low']
    open_close_return = (df['close'] - df['open']) / df['open']
    intraday_momentum = 0.5 * (high_low_diff + open_close_return)

    # Volume Shock Filter
    df['vol_avg_5'] = df['volume'].rolling(window=5).mean()
    volume_ratio = df['volume'] / df['vol_avg_5']
    intraday_momentum = intraday_momentum * (volume_ratio > 1.8)

    # Trend Confirmation with EMA
    df['short_ema'] = df['close'].ewm(span=5, adjust=False).mean()
    df['long_ema'] = df['close'].ewm(span=20, adjust=False).mean()
    crossover_signal = (df['short_ema'] > df['long_ema']).astype(int) - (df['short_ema'] < df['long_ema']).astype(int)

    # Integrate with Intraday Momentum
    intraday_momentum = intraday_momentum * (1 + 0.5 * crossover_signal)

    # Calculate On-Balance Volume (OBV)
    df['obv'] = 0
    df['obv'] = df.apply(lambda row: row['obv'] + row['volume'] if row['close'] > df.shift(1)['close'] 
                         else (row['obv'] - row['volume'] if row['close'] < df.shift(1)['close'] else row['obv']), axis=1)

    # Weight OBV by EMA Difference
    ema_diff = abs(df['short_ema'] - df['long_ema'])
    final_obv_factor = df['obv'] * ema_diff

    # Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Apply RSI Condition
    rsi_adjustment = np.where(rsi < 30, 1.2, np.where(rsi > 70, 0.8, 1))

    # Final Alpha Factor
    final_alpha_factor = intraday_momentum * final_obv_factor * rsi_adjustment

    return final_alpha_factor
