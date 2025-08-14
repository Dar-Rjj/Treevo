import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=10, ema_span=5):
    # Calculate Volume-Weighted Average Price (VWAP)
    df['TypicalPrice'] = (df['High'] + df['Low']) / 2
    df['VWAP'] = (df['TypicalPrice'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Calculate Daily Returns using Close price
    df['DailyReturnClose'] = (df['Close'] - df['VWAP'].shift(1)) / df['VWAP'].shift(1)
    
    # Calculate Daily Returns using High price
    df['DailyReturnHigh'] = (df['High'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # Aggregate Volume-Weighted Momentum
    df['ProductClose'] = df['DailyReturnClose'] * df['Volume']
    df['ProductHigh'] = df['DailyReturnHigh'] * df['Volume']
    
    # Sum the Products and Aggregate Volume over N days
    df['AggregateProduct'] = df['ProductClose'].rolling(window=N).sum() + df['ProductHigh'].rolling(window=N).sum()
    df['AggregateVolume'] = df['Volume'].rolling(window=N).sum()
    
    # Final VWAM Calculation
    df['VWAM'] = df['AggregateProduct'] / df['AggregateVolume']
    
    # Smooth the Daily Return with Exponential Moving Average (EMA)
    df['SmoothedReturn'] = df['DailyReturnClose'].ewm(span=ema_span, adjust=False).mean()
    
    # Multiply Smoothed Return by Volume
    df['AlphaFactor'] = df['SmoothedReturn'] * df['Volume']
    
    return df['AlphaFactor']
