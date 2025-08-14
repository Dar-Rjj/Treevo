import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Raw Momentum
    df['RawMomentum'] = df['close'] - df['close'].shift(20)
    
    # Adjust for Volume
    df['AvgVol'] = df['volume'].rolling(window=20).mean()
    df['VolRatio'] = df['volume'] / df['AvgVol']
    df['AdjMomentum'] = df['RawMomentum'] * df['VolRatio']
    
    # Incorporate Enhanced Price Gaps
    df['GapOC'] = df['open'] - df['close']
    df['GapHL'] = df['high'] - df['low']
    df['CombinedMomentum'] = df['AdjMomentum'] + df['GapOC'] + df['GapHL']
    
    # Confirm with Volume
    df['VolMA5'] = df['volume'].rolling(window=5).mean()
    df['VolMA20'] = df['volume'].rolling(window=20).mean()
    df['ConfirmedMomentum'] = df['CombinedMomentum'].where(df['VolMA5'] > df['VolMA20'], df['CombinedMomentum'] * 0.5)
    
    # Adjust Momentum by ATR
    df['TrueRange'] = df[['high', 'low', 'close']].apply(lambda x: max(abs(x[0] - x[1]), abs(x[0] - x[2].shift(1)), abs(x[1] - x[2].shift(1))), axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    df['AdjMomentumATR'] = df['ConfirmedMomentum'] / df['ATR']
    
    # Introduce Final Volume Adjustment
    df['FinalFactor'] = df['AdjMomentumATR'] * df['VolMA20']
    
    # Calculate Close-to-Low and Close-to-High Distances
    df['CloseToLow'] = df['close'] - df['low']
    df['CloseToHigh'] = df['high'] - df['close']
    
    # Adjust for Volume
    df['VolumeRatio'] = df['volume'] / df['VolMA20']
    df['VolumeAdjustedFactor'] = df['FinalFactor'] * df['VolumeRatio']
    
    # Combine Factors with Close-to-Low and Close-to-High Distances
    df['CombinedFactor1'] = df['VolumeAdjustedFactor'] * df['CloseToLow']
    df['CombinedFactor2'] = df['VolumeAdjustedFactor'] * df['CloseToHigh']
    
    # Final Factor
    df['FinalCombinedFactor'] = df['CombinedFactor1'] + df['CombinedFactor2']
    df['FinalAlpha'] = df['FinalCombinedFactor']
    
    return df['FinalAlpha']
