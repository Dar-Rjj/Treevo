import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Thought 1: Analyze price patterns to identify bullish or bearish trends
    short_term_ma = df['close'].rolling(window=5).mean()
    long_term_ma = df['close'].rolling(window=20).mean()
    ma_cross_over = (short_term_ma > long_term_ma) & (short_term_ma.shift(1) <= long_term_ma.shift(1))
    ma_cross_under = (short_term_ma < long_term_ma) & (short_term_ma.shift(1) >= long_term_ma.shift(1))

    # Thought 1.2: Identify trend strength and direction using the Average Directional Index (ADX)
    def calculate_tr(high, low, close_prev):
        return max(high - low, abs(high - close_prev), abs(low - close_prev))

    tr = df.apply(lambda x: calculate_tr(x['high'], x['low'], df.loc[x.name - pd.Timedelta(days=1), 'close']), axis=1)
    atr = tr.rolling(window=14).mean()

    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    plus_dm = up_move.where(up_move > down_move, 0)
    minus_dm = down_move.where(down_move > up_move, 0)

    plus_di = 100 * (plus_dm.rolling(window=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).sum() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=14).mean()

    # Thought 2: Examine trading volume to infer market interest and momentum
    obv = df['volume'].copy()
    obv[0] = 0
    for i in range(1, len(obv)):
        if df['close'][i] > df['close'][i-1]:
            obv[i] = obv[i-1] + df['volume'][i]
        elif df['close'][i] < df['close'][i-1]:
            obv[i] = obv[i-1] - df['volume'][i]
        else:
            obv[i] = obv[i-1]

    pvt = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']
    pvt = pvt.cumsum()

    # Thought 3: Measure volatility to assess market uncertainty
    tr = df.apply(lambda x: calculate_tr(x['high'], x['low'], df.loc[x.name - pd.Timedelta(days=1), 'close']), axis=1)
    atr = tr.rolling(window=14).mean()

    sma = df['close'].rolling(window=20).mean()
    std_dev = df['close'].rolling(window=20).std()
    upper_bb = sma + 2 * std_dev
    lower_bb = sma - 2 * std_dev
    bb_width = (upper_bb - lower_bb) / sma

    # Thought 4: Evaluate the relationship between opening and closing prices
    oc_spread = df['close'] - df['open']
    co_ratio = df['close'] / df['open']

    # Thought 5: Combine multiple factors for a more comprehensive analysis
    factor = (ma_cross_over.astype(int) * 0.3 +
              (adx > 25).astype(int) * 0.2 +
              (pvt > pvt.rolling(window=14).mean()).astype(int) * 0.1 +
              (bb_width > bb_width.rolling(window=14).mean()).astype(int) * 0.1 +
              (oc_spread > 0).astype(int) * 0.3)

    return factor
