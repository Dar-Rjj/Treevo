import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data: pd.DataFrame) -> pd.Series:
    # Momentum Factors
    data['50_day_SMA'] = data['close'].rolling(window=50).mean()
    data['200_day_SMA'] = data['close'].rolling(window=200).mean()
    data['10_day_ROC'] = data['close'].pct_change(periods=10)
    data['30_day_ROC'] = data['close'].pct_change(periods=30)

    # Volatility Factors
    log_prices = np.log(data['close'])
    data['20_day_HV'] = log_prices.rolling(window=20).std() * np.sqrt(20)
    data['60_day_HV'] = log_prices.rolling(window=60).std() * np.sqrt(60)
    tr14 = pd.DataFrame({'hl': data['high'] - data['low'],
                         'hpc': (data['high'] - data['close'].shift(1)).abs(),
                         'lpc': (data['low'] - data['close'].shift(1)).abs()}).max(axis=1)
    data['14_day_ATR'] = tr14.rolling(window=14).mean()

    # Volume Factors
    data['10_day_VROC'] = data['volume'].pct_change(periods=10)
    data['30_day_VROC'] = data['volume'].pct_change(periods=30)
    obv = [0]
    for i in range(1, len(data)):
        if data.iloc[i]['close'] > data.iloc[i-1]['close']:
            obv.append(obv[-1] + data.iloc[i]['volume'])
        elif data.iloc[i]['close'] < data.iloc[i-1]['close']:
            obv.append(obv[-1] - data.iloc[i]['volume'])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv

    # Technical Indicators
    delta = data["close"].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['14_day_RSI'] = 100 - (100 / (1 + rs))

    exp12 = data['close'].ewm(span=12, adjust=False).mean()
    exp26 = data['close'].ewm(span=26, adjust=False).mean()
    macd_line = exp12 - exp26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = macd_line - signal_line

    # Combine factors to a single alpha factor
    alpha_factor = (data['50_day_SMA'] - data['200_day_SMA']) + \
                   data['10_day_ROC'] - data['30_day_ROC'] + \
                   data['20_day_HV'] - data['60_day_HV'] + \
                   data['14_day_ATR'] + \
                   data['10_day_VROC'] - data['30_day_VROC'] + \
                   data['OBV'] + \
                   data['14_day_RSI'] + \
                   data['MACD_Histogram']

    return alpha_factor
