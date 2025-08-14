import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily log returns using close price
    df['log_return'] = np.log(df['close']).diff()
    
    # Initialize short-term and long-term periods
    short_term_period = 10
    long_term_period = 30
    
    # Calculate the weighted moving average for short-term and long-term
    def calculate_wma(series, weights):
        return (series * weights).sum() / weights.sum()
    
    # Short-Term Moving Average (SMA)
    sma = df['log_return'].rolling(window=short_term_period).apply(
        lambda x: calculate_wma(x, df['volume'].iloc[x.index[0]:x.index[-1]+1]),
        raw=False
    )
    
    # Long-Term Moving Average (LMA)
    lma = df['log_return'].rolling(window=long_term_period).apply(
        lambda x: calculate_wma(x, df['volume'].iloc[x.index[0]:x.index[-1]+1]),
        raw=False
    )
    
    # Generate alpha factor by subtracting LMA from SMA
    alpha_factor = sma - lma
    
    return alpha_factor
