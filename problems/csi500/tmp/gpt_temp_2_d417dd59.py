import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['Total Volume'] = df['volume']
    df['Total Dollar Value'] = df['volume'] * df['close']
    df['VWAP'] = df.groupby(df.index.date)['Total Dollar Value'].transform('sum') / df.groupby(df.index.date)['Total Volume'].transform('sum')
    
    # Calculate VWAP Deviation
    df['VWAP Deviation'] = df['close'] - df['VWAP']
    
    # Calculate Cumulative VWAP Deviation
    df['Cumulative VWAP Deviation'] = df['VWAP Deviation'].cumsum()
    
    # Integrate Adaptive Exponential Moving Average (EMA)
    short_term_ema_period = 5
    medium_term_ema_period = 10
    df['Short-Term EMA of VWAP'] = df['VWAP'].ewm(span=short_term_ema_period, adjust=False).mean()
    df['Medium-Term EMA of VWAP'] = df['VWAP'].ewm(span=medium_term_ema_period, adjust=False).mean()
    df['Dynamic Threshold'] = (df['Short-Term EMA of VWAP'] + df['Medium-Term EMA of VWAP']) / 2
    
    # Integrate Multi-Period Momentum
    short_term_momentum_period = 5
    medium_term_momentum_period = 10
    df['Short-Term Momentum'] = df['close'].pct_change(short_term_momentum_period)
    df['Medium-Term Momentum'] = df['close'].pct_change(medium_term_momentum_period)
    df['Combined Momentum'] = (df['Short-Term Momentum'] + df['Medium-Term Momentum']) / 2
    
    # Integrate Volatility
    df['Returns'] = df['close'].pct_change()
    df['Short-Term Volatility'] = df['Returns'].rolling(window=short_term_momentum_period).std()
    df['Medium-Term Volatility'] = df['Returns'].rolling(window=medium_term_momentum_period).std()
    df['Combined Volatility'] = (df['Short-Term Volatility'] + df['Medium-Term Volatility']) / 2
    
    # Incorporate Volume-Adjusted Returns
    df['Volume-Adjusted Return'] = (df['close'] - df['open']) / df['volume']
    df['Prev Volume-Adjusted Return'] = df['Volume-Adjusted Return'].shift(1)
    
    # Final Alpha Factor
    df['Alpha Factor'] = (
        df['Cumulative VWAP Deviation'] +
        df['Dynamic Threshold'] +
        df['Combined Momentum'] +
        df['Combined Volatility'] +
        df['Prev Volume-Adjusted Return']
    )
    
    return df['Alpha Factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
