import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Price Spread
    df['HighLowSpread'] = df['high'] - df['low']
    
    # Compute 20-Day Moving Average of High-Low Spread
    df['HighLowSpread_20_MA'] = df['HighLowSpread'].rolling(window=20).mean()
    
    # Calculate Cumulative Return Over 20 Days
    df['CumulativeReturn'] = (df['close'] / df['close'].shift(20)) - 1
    
    # Calculate Volume Variance
    df['VolumeVariance'] = df['volume'].rolling(window=20).var()
    
    # Adjust for Volume Variance
    df['VolumeVarianceInverse'] = 1 / df['VolumeVariance']
    df['AdjustedCumulativeReturn'] = df['CumulativeReturn'] * df['VolumeVarianceInverse']
    
    # Calculate Volume Weighted Close Price
    df['VolumeWeightedClose'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Calculate Volume-Weighted Momentum
    df['VolumeWeightedClose_20_SMA'] = df['VolumeWeightedClose'].rolling(window=20).mean()
    df['VolumeWeightedMomentum'] = df['VolumeWeightedClose'] - df['VolumeWeightedClose_20_SMA']
    
    # Incorporate Price Variability
    df['TrueRange'] = df.apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    df['TrueRange_20_MA'] = df['TrueRange'].rolling(window=20).mean()
    df['AdjustedMomentum'] = df['VolumeWeightedMomentum'] - df['TrueRange_20_MA']
    
    # Combine Adjusted Cumulative Return and Adjusted Momentum
    df['FinalAlphaFactor'] = df['AdjustedCumulativeReturn'] + df['AdjustedMomentum']
    
    # Include only if today's volume > 20-day average volume
    df['FinalAlphaFactor'] = df['FinalAlphaFactor'].where(df['volume'] > df['volume'].rolling(window=20).mean(), other=pd.NA)
    
    return df['FinalAlphaFactor'].dropna()
