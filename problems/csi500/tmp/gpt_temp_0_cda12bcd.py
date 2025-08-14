import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['volume'] * df['close']
    df['vwap'] = df.groupby(df.index.date)['total_dollar_value'].transform('sum') / df.groupby(df.index.date)['total_volume'].transform('sum')
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']
    
    # Calculate Exponential Moving Average (EMA) of VWAP
    df['vwap_ema'] = df['vwap'].ewm(span=10, adjust=False).mean()
    
    # Calculate VWAP EMA Deviation
    df['vwap_ema_deviation'] = df['vwap'] - df['vwap_ema']
    
    # Calculate Volume Trend
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Calculate Multi-Period Momentum
    df['momentum'] = df['close'].pct_change().rolling(window=20).sum()
    
    # Calculate Price Volatility (Average True Range)
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1)
    df['average_true_range'] = df['true_range'].rolling(window=14).mean()
    
    # Integrate Factors into Final Alpha Signal
    df['alpha_signal'] = (
        df['vwap_deviation'] + 
        df['vwap_ema_deviation'] + 
        df['volume_change'] + 
        df['momentum'] + 
        df['average_true_range']
    )
    
    return df['alpha_signal']

# Example usage:
# data = pd.DataFrame({
#     'open': [100, 101, 99, 102],
#     'high': [102, 103, 105, 106],
#     'low': [98, 99, 97, 100],
#     'close': [101, 102, 100, 105],
#     'amount': [1000, 1500, 2000, 2500],
#     'volume': [100, 150, 200, 250]
# }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']))

# alpha_signal = heuristics_v2(data)
# print(alpha_signal)
