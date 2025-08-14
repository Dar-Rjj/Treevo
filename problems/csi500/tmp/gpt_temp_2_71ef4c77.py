import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['TotalVolume'] = df['volume']
    df['TotalDollarValue'] = df['volume'] * df['close']
    df['VWAP'] = df.groupby(df.index.date)['TotalDollarValue'].cumsum() / df.groupby(df.index.date)['TotalVolume'].cumsum()
    
    # Calculate VWAP Deviation
    df['VWAPDeviation'] = df['close'] - df['VWAP']
    
    # Calculate Exponential Moving Average (EMA) of VWAP with Adaptive Smoothing Factor
    def adaptive_ema(series, span=10, alpha_adjustment=0.5):
        ema = series.ewm(span=span, adjust=False).mean()
        return ema + alpha_adjustment * (series - ema)
    
    df['VWAP_EMA'] = adaptive_ema(df['VWAP'])
    df['VWAP_EMADeviation'] = df['VWAP'] - df['VWAP_EMA']
    
    # Calculate Cumulative VWAP Deviation
    df['CumulativeVWAPDeviation'] = df['VWAPDeviation'].rolling(window=20).sum()
    
    # Calculate Volume Trend
    df['VolumeChange'] = df['volume'] - df['volume'].shift(1)
    
    # Calculate Multi-Period Momentum
    df['Momentum'] = df['close'].pct_change().rolling(window=10).sum()
    
    # Calculate True Range
    df['HighLowDiff'] = df['high'] - df['low']
    df['HighCloseDiff'] = (df['high'] - df['close'].shift(1)).abs()
    df['LowCloseDiff'] = (df['low'] - df['close'].shift(1)).abs()
    df['TrueRange'] = df[['HighLowDiff', 'HighCloseDiff', 'LowCloseDiff']].max(axis=1)
    
    # Calculate Average True Range (ATR)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    
    # Calculate Exponential Moving Average (EMA) of ATR with Adaptive Smoothing Factor
    df['ATR_EMA'] = adaptive_ema(df['ATR'])
    
    # Integrate Factors into Final Alpha Signal
    df['AlphaSignal'] = (
        0.3 * df['CumulativeVWAPDeviation'] +
        0.2 * df['VWAP_EMADeviation'] +
        0.1 * df['VolumeChange'] +
        0.2 * df['Momentum'] +
        0.2 * (df['ATR_EMA'] - df['ATR'])
    )
    
    return df['AlphaSignal']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_signal = heuristics_v2(df)
# print(alpha_signal)
