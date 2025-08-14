import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df, n=20, m=60):
    # Momentum-based Alpha Factors
    df['PriceMomentum'] = df['close'] / df['close'].shift(n)
    df['VolumeMomentum'] = df['volume'] / df['volume'].shift(n)
    df['AmountMomentum'] = df['amount'] / df['amount'].shift(n)
    
    # Trend-following Alpha Factors
    df['ShortTermMA'] = df['close'].rolling(window=n).mean()
    df['LongTermMA'] = df['close'].rolling(window=m).mean()
    df['MACrossover'] = df['ShortTermMA'] - df['LongTermMA']
    
    df['ShortTermMA_Amount'] = df['amount'].rolling(window=n).mean()
    df['LongTermMA_Amount'] = df['amount'].rolling(window=m).mean()
    df['MACrossover_Amount'] = df['ShortTermMA_Amount'] - df['LongTermMA_Amount']
    
    df['ShortTermMA_Volume'] = df['volume'].rolling(window=n).mean()
    df['LongTermMA_Volume'] = df['volume'].rolling(window=m).mean()
    df['MACrossover_Volume'] = df['ShortTermMA_Volume'] - df['LongTermMA_Volume']
    
    # Volatility-based Alpha Factors
    df['DailyReturn'] = df['close'].pct_change()
    df['HistoricalVolatility'] = df['DailyReturn'].rolling(window=n).std()
    
    df['HighLowVolatility'] = (df['high'] - df['low']).rolling(window=n).mean()
    df['RangeVolatility'] = (df['high'] - df['low']) / df['close']
    df['AverageRangeVolatility'] = df['RangeVolatility'].rolling(window=n).mean()
    
    # Reversal-based Alpha Factors
    df['ShortTermReversal'] = (df['high'].rolling(window=n).max() - df['low'].rolling(window=n).min()) / df['close']
    df['LongTermReversal'] = (df['high'].rolling(window=m).max() - df['low'].rolling(window=m).min()) / df['close']
    
    # Liquidity-based Alpha Factors
    df['DollarVolume'] = df['volume'] * df['close']
    df['AmihudIlliquidity'] = (df['DailyReturn'].abs() / df['DollarVolume']).rolling(window=n).mean()
    
    # Drop intermediate columns
    df.drop(columns=['ShortTermMA', 'LongTermMA', 'ShortTermMA_Amount', 'LongTermMA_Amount', 
                     'ShortTermMA_Volume', 'LongTermMA_Volume', 'DailyReturn', 'RangeVolatility'], inplace=True)
    
    return df
