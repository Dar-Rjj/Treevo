import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics(df):
    # Calculate Enhanced Intraday Momentum
    high_low_diff = (df['high'] - df['low']) / df['open']
    open_close_return = (df['close'] - df['open']) / df['open']
    intraday_momentum = (high_low_diff + open_close_return) / 2
    intraday_momentum = intraday_momentum ** 2  # Non-Linear Transformation

    # Apply Synchronized Volume and Amount Shock Filter
    volume_change = df['volume'] - df['volume'].shift(1)
    volume_ratio = df['volume'] / df['volume'].shift(1)
    amount_change = df['amount'] - df['amount'].shift(1)
    amount_ratio = df['amount'] / df['amount'].shift(1)

    volume_threshold = 1.5
    amount_threshold = 1.5

    volume_shock = (volume_ratio > volume_threshold) & (amount_ratio > amount_threshold)
    volume_ema = volume_ratio.ewm(alpha=0.2).mean()
    amount_ema = amount_ratio.ewm(alpha=0.2).mean()

    # Synchronize Volume and Price Changes
    price_change = df['close'] - df['close'].shift(1)
    synchronized_changes = np.sign(volume_change * price_change)

    # EMA Cross Signal
    short_ema = df['close'].ewm(span=5, adjust=False).mean()
    long_ema = df['close'].ewm(span=20, adjust=False).mean()
    momentum_signal = (short_ema > long_ema).astype(int) * 2 - 1  # 1 for Bullish, -1 for Bearish

    # Calculate On-Balance Volume (OBV)
    obv = pd.Series(0, index=df.index)
    obv[1:] = (np.sign(df['close'].diff().dropna()) * df['volume']).fillna(0).cumsum()

    # Weight OBV by the Difference between Short and Long EMA
    ema_diff_weight = np.abs(short_ema - long_ema)
    final_obv_factor = obv * ema_diff_weight

    # Combine Enhanced Intraday Momentum, Synchronized Volume, and OBV-Momentum Factor
    final_alpha_factor = intraday_momentum * synchronized_changes * final_obv_factor

    return final_alpha_factor
