import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Short-Term Volume Adjusted Return
    short_term_window = 5
    df['ShortTermReturn'] = df['close'].pct_change()
    df['ShortTermVolumeAdjustedReturn'] = df['ShortTermReturn'] * df['volume']
    df['ShortTermVolumeAdjustedReturn'] = df['ShortTermVolumeAdjustedReturn'].rolling(window=short_term_window).mean()

    # Calculate Long-Term Volume Adjusted Return
    long_term_window = 20
    df['LongTermReturn'] = df['close'].pct_change()
    df['LongTermVolumeAdjustedReturn'] = df['LongTermReturn'] * df['volume']
    df['LongTermVolumeAdjustedReturn'] = df['LongTermVolumeAdjustedReturn'].rolling(window=long_term_window).mean()

    # Calculate Price Reversal Indicator
    def identify_local_extrema(series, window=10):
        local_highs = (series.rolling(window=window, center=True).max() == series)
        local_lows = (series.rolling(window=window, center=True).min() == series)
        return local_highs, local_lows

    high_peaks, low_valleys = identify_local_extrema(df['high']), identify_local_extrema(df['low'])
    df['TimeSinceLastHigh'] = (~high_peaks).cumsum()
    df['TimeSinceLastLow'] = (~low_valleys).cumsum()
    df['PriceReversalIndicator'] = (df['TimeSinceLastHigh'] - df['TimeSinceLastLow']) / (df['TimeSinceLastHigh'] + df['TimeSinceLastLow'])

    # Calculate Intraday High-Low Spread
    df['IntradaySpread'] = df['high'] - df['low']

    # Adjust Intraday High-Low Spread by Volume
    ema_span = 10
    df['VolumeEMA'] = df['volume'].ewm(span=ema_span).mean()
    df['AdjustedVolume'] = df['volume'] / df['VolumeEMA']
    df['AdjustedIntradaySpread'] = df['IntradaySpread'] * df['AdjustedVolume']

    # Further Adjustment by Open Price Volatility
    df['OpenReturns'] = df['open'].pct_change()
    open_volatility = df['OpenReturns'].rolling(window=20).std()
    df['AdjustedIntradaySpreadVolatility'] = df['AdjustedIntradaySpread'] / open_volatility

    # Calculate Relative Strength Indicator (RSI)
    def rsi(close, window=14):
        delta = close.diff(1)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        avg_gain = up.rolling(window=window).mean()
        avg_loss = abs(down.rolling(window=window).mean())
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI'] = rsi(df['close'])

    # Aggregate Metrics
    df['CombinedReturns'] = df['ShortTermVolumeAdjustedReturn'] - df['LongTermVolumeAdjustedReturn']
    df['CombinedFactor'] = (df['CombinedReturns'] + df['PriceReversalIndicator'] + df['AdjustedIntradaySpreadVolatility']) * df['volume']

    # Final Alpha Factor
    df['AlphaFactor'] = (df['CombinedFactor'] + df['RSI']).fillna(0)

    return df['AlphaFactor']
