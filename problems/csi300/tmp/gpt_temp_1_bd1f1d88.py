import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    df['HighLowSpread'] = df['high'] - df['low']
    
    # Calculate Intraday Return
    df['IntradayReturn'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Intraday Range
    df['IntradayRange'] = (df['high'] - df['low']) / df['open']
    
    # Calculate Intraday Volatility (Average True Range over a period)
    df['TrueRange'] = df[['high' - 'low', 
                          'high' - df['close'].shift(1), 
                          df['close'].shift(1) - df['low']]].max(axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    
    # Calculate Intraday Momentum
    df['IntradayMomentum'] = df['HighLowSpread'] - df['HighLowSpread'].shift(1)
    
    # Calculate Intraday Reversal
    df['PrevCloseOpenDiff'] = df['close'].shift(1) - df['open'].shift(1)
    df['IntradayReversal'] = (df['close'] - df['open'] - df['PrevCloseOpenDiff']) * df['IntradayMomentum']
    
    # Calculate Volume Trend (Exponential Moving Average of Volume)
    df['VolumeEMA'] = df['volume'].ewm(span=14, adjust=False).mean()
    
    # Compute Volume Reversal Component
    df['VolumeReversalComponent'] = df.apply(lambda row: row['IntradayReturn'] if row['volume'] > row['VolumeEMA'] else -row['IntradayReturn'], axis=1)
    
    # Introduce Intraday Gap Factor
    df['PrevClose'] = df['close'].shift(1)
    df['IntradayGapFactor'] = (df['open'] - df['PrevClose']) / df['PrevClose']
    
    # Incorporate Transaction Amount
    df['AmountSMA'] = df['amount'].rolling(window=14).mean()
    df['TransactionAmountComponent'] = df.apply(lambda row: row['IntradayReturn'] if row['amount'] > row['AmountSMA'] else 0, axis=1)
    
    # Combine All Components
    df['AlphaFactor'] = (df['IntradayReturn'] * df['IntradayRange'] * df['ATR'] + 
                         df['IntradayReversal'] + 
                         df['VolumeReversalComponent'] + 
                         df['IntradayGapFactor'] + 
                         df['TransactionAmountComponent'])
    
    return df['AlphaFactor']
