import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    df['IntradayReturn'] = (df['high'] - df['low']) / df['open']
    
    # Smooth Intraday Return
    df['EMA_IntradayReturn'] = df['IntradayReturn'].ewm(span=5).mean()
    df['SmoothedIntradayReturn'] = df['IntradayReturn'] - df['EMA_IntradayReturn']
    
    # Incorporate Volume Intensity
    df['AvgVolume'] = df['volume'].rolling(window=5).mean()
    df['IntradayMomentum'] = df['SmoothedIntradayReturn'] * (1 / df['AvgVolume'])
    
    # Detect Volume Spikes
    df['PrevDayVolume'] = df['volume'].shift(1)
    df['VolumeChange'] = df['volume'] - df['PrevDayVolume']
    df['AvgVolumeChange'] = df['VolumeChange'].rolling(window=25).mean()
    df['Spike'] = (df['VolumeChange'] > 1.75 * df['AvgVolumeChange']).astype(int)
    
    # Combine Intraday Momentum and Volume Impact
    df['AdjustedIntradayMomentum'] = df['IntradayMomentum'] * (0.5 if df['Spike'] else 1)
    
    # Compute Daily Price Change
    df['DailyPriceChange'] = df['close'] - df['close'].shift(1)
    
    # Compute Smoothed Price Momentum
    df['EMA_DailyPriceChange'] = df['DailyPriceChange'].ewm(span=5).mean()
    df['SmoothedPriceMomentum'] = df['DailyPriceChange'] - df['EMA_DailyPriceChange']
    
    # Adjust Momentum by Volume Spike
    df['Momentum'] = df['EMA_DailyPriceChange'] if df['Spike'] else df['DailyPriceChange']
    
    # Calculate Weighted Price Movement
    df['WeightedPriceMovement'] = (df['close'] - df['open']) * df['volume'] / df['AvgVolume']
    
    # Smooth Close Price Change
    df['ClosePriceChange'] = df['close'] - df['close'].shift(1)
    df['EMA_ClosePriceChange'] = df['ClosePriceChange'].ewm(span=5).mean()
    df['SmoothedClosePriceChange'] = df['ClosePriceChange'] - df['EMA_ClosePriceChange']
    
    # Introduce Reversal Indicator
    df['LongTermMomentum'] = df['DailyPriceChange'].rolling(window=20).mean()
    df['ShortTermMomentum'] = df['DailyPriceChange'].rolling(window=5).mean()
    df['Reversal'] = df['ShortTermMomentum'] - df['LongTermMomentum']
    
    # Final Alpha Signal
    df['AlphaSignal'] = (
        df['AdjustedIntradayMomentum'] * df['SmoothedPriceMomentum'] +
        df['WeightedPriceMovement'] + 
        df['SmoothedClosePriceChange'] + 
        df['Reversal']
    )
    
    return df['AlphaSignal'].dropna()
