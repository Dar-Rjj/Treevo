import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Weighted Average Price (VWAP)
    df['TypicalPrice'] = (df['High'] + df['Low']) / 2
    df['VWAP'] = (df['TypicalPrice'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    
    # Calculate Daily Return
    df['DailyReturn'] = (df['Close'] - df['VWAP']) / df['VWAP']
    
    # Smooth and Scale the Daily Return
    span_ema = 10
    df['SmoothedReturn'] = df['DailyReturn'].ewm(span=span_ema).mean()
    df['ScaledReturn'] = df['SmoothedReturn'] * df['Volume']
    
    # Calculate High-to-Low Price Range
    df['Range'] = df['High'] - df['Low']
    
    # Calculate Open-Adjusted Range
    df['HighOpenDiff'] = df['High'] - df['Open']
    df['OpenLowDiff'] = df['Open'] - df['Low']
    df['OpenAdjustedRange'] = df[['HighOpenDiff', 'OpenLowDiff']].max(axis=1)
    
    # Calculate Enhanced Price Momentum with Volume Adjustment
    lookback_period = 10
    df['EMA_Close'] = df['Close'].ewm(span=lookback_period).mean()
    df['PriceDifference'] = df['Close'] - df['EMA_Close']
    df['MomentumScore'] = df['PriceDifference'] / df['EMA_Close']
    df['CumulativeVolume'] = df['Volume'].rolling(window=lookback_period).sum()
    df['AdjustedMomentumScore'] = df['MomentumScore'] * df['CumulativeVolume']
    
    # Calculate Trading Intensity
    df['VolumeChange'] = df['Volume'].diff()
    df['AmountChange'] = df['amount'].diff()
    df['TradingIntensity'] = df['VolumeChange'] / df['AmountChange']
    
    # Weight the Range by Trading Intensity
    scaling_factor = 1500
    df['WeightedRange'] = df['Range'] * (df['TradingIntensity'] * scaling_factor)
    
    # Combine Momentum and Weighted Range
    df['CombinedFactor'] = df['ScaledReturn'] + df['OpenAdjustedRange'] + df['AdjustedMomentumScore'] + df['WeightedRange']
    
    # Additional Features
    df['CloseDiff'] = df['Close'].diff()
    df['AvgOpenClose'] = (df['Open'] + df['Close']) / 2
    
    # Calculate the Relative Strength Index (RSI)
    rsi_period = 14
    df['Gain'] = df['Close'].diff().clip(lower=0).fillna(0)
    df['Loss'] = -df['Close'].diff().clip(upper=0).fillna(0)
    df['AverageGain'] = df['Gain'].rolling(window=rsi_period).mean()
    df['AverageLoss'] = df['Loss'].rolling(window=rsi_period).mean()
    df['RS'] = df['AverageGain'] / df['AverageLoss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    
    # Calculate the On-Balance Volume (OBV)
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    
    # Combine All Alpha Factors
    df['AlphaFactor'] = df['CombinedFactor'] + (df['VWAP'] * df['SmoothedReturn']) + df['RSI'] + df['OBV']
    
    return df['AlphaFactor']
