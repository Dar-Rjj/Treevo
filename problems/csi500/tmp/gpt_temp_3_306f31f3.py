import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['close'] * df['volume']
    df['vwap'] = df.groupby(df.index.date)['total_dollar_value'].transform('sum') / df.groupby(df.index.date)['total_volume'].transform('sum')
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']
    
    # Calculate Exponential Moving Average (EMA) of VWAP
    smoothing_factor = 2 / (len(df) + 1)
    df['vwap_ema'] = df['vwap'].ewm(alpha=smoothing_factor, adjust=False).mean()
    
    # Calculate VWAP EMA Deviation
    df['vwap_ema_deviation'] = df['vwap'] - df['vwap_ema']
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].rolling(window=20).sum()
    
    # Calculate Volume Trend
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Calculate Multi-Period Momentum
    df['momentum'] = df['close'].pct_change().rolling(window=20).sum()
    
    # Calculate True Range
    df['high_low_diff'] = df['high'] - df['low']
    df['high_prev_close_diff'] = abs(df['high'] - df['close'].shift(1))
    df['low_prev_close_diff'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low_diff', 'high_prev_close_diff', 'low_prev_close_diff']].max(axis=1)
    
    # Calculate Average True Range (ATR)
    df['atr'] = df['true_range'].rolling(window=20).mean()
    
    # Calculate Cumulative ATR
    df['cumulative_atr'] = df['atr'].rolling(window=20).sum()
    
    # Integrate Factors into Final Alpha Signal
    df['alpha_signal'] = (
        0.3 * df['cumulative_vwap_deviation'] +
        0.2 * df['vwap_ema_deviation'] +
        0.1 * df['volume_change'] +
        0.2 * df['momentum'] +
        0.2 * df['cumulative_atr']
    )
    
    return df['alpha_signal'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col='date')
# alpha_signal = heuristics_v2(df)
# print(alpha_signal)
