import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Simple Momentum
    df['SimpleMomentum'] = df['close'] - df['close'].shift(20)
    
    # Adjust for Volume
    df['AvgVol'] = df['volume'].rolling(window=20).mean()
    df['VolRatio'] = df['volume'] / df['AvgVol']
    df['AdjMomentum'] = df['SimpleMomentum'] * df['VolRatio']
    
    # Incorporate Enhanced Price Gaps
    df['GapOC'] = df['open'] - df['close']
    df['GapHL'] = df['high'] - df['low']
    df['CombinedMomentum'] = df['AdjMomentum'] + df['GapOC'] + df['GapHL']
    
    # Confirm with Volume
    df['VolMA5'] = df['volume'].rolling(window=5).mean()
    df['VolMA20'] = df['volume'].rolling(window=20).mean()
    df['ConfirmedMomentum'] = df['CombinedMomentum'] * (df['VolMA5'] > df['VolMA20']).astype(int) * 1.0
    df['ConfirmedMomentum'] += df['CombinedMomentum'] * (df['VolMA5'] <= df['VolMA20']).astype(int) * 0.5
    
    # Adjust Momentum by ATR
    df['TR'] = df[['high', 'low']].apply(lambda x: max(x['high'] - x['low'], 
                                                        abs(x['high'] - df['close'].shift(1)), 
                                                        abs(x['low'] - df['close'].shift(1))), axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['AdjMomentumATR'] = df['ConfirmedMomentum'] / df['ATR']
    
    # Measure Volume Impact
    df['VolEMA5'] = df['volume'].ewm(span=5).mean()
    df['VolumeRatio'] = df['volume'] / df['VolEMA5']
    df['VolumeAdjustedFactor'] = df['AdjMomentumATR'] * df['VolumeRatio']
    
    # Calculate Close-to-High and Close-to-Low Distances
    df['CloseToHigh'] = df['high'] - df['close']
    df['CloseToLow'] = df['close'] - df['low']
    
    # Combine Factors with Close-to-Low and Close-to-High Distances
    df['CombinedFactor1'] = df['VolumeAdjustedFactor'] * df['CloseToLow']
    df['CombinedFactor2'] = df['VolumeAdjustedFactor'] * df['CloseToHigh']
    df['FinalCombinedFactor'] = df['CombinedFactor1'] + df['CombinedFactor2']
    
    # Enhanced Price Reversal Sensitivity
    df['HCSpread'] = df['high'] - df['close']
    df['OLSpread'] = df['open'] - df['low']
    df['WeightedHC'] = df['HCSpread'] * df['volume']
    df['WeightedOL'] = df['OLSpread'] * df['volume']
    df['TotalWeightedSpreads'] = df['WeightedHC'] + df['WeightedOL']
    
    # Final Alpha Factor
    df['FinalAlpha'] = df['FinalCombinedFactor'] - df['TotalWeightedSpreads']
    
    return df['FinalAlpha']
