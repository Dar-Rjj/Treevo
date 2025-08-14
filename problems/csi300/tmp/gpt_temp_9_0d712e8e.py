import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Historical Volume-Weighted Close
    df['VolumeWeightedClose'] = (df['volume'] * df['close']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Raw Weighted Momentum
    df['RawMomentum'] = df['close'] - df['VolumeWeightedClose']
    
    # Adjust for Volume
    df['AvgVol'] = df['volume'].rolling(window=20).mean()
    df['VolRatio'] = df['volume'] / df['AvgVol']
    df['AdjMomentum'] = df['RawMomentum'] * df['VolRatio']
    
    # Incorporate Price Gaps
    df['GapOC'] = df['open'] - df['close']
    df['GapHL'] = df['high'] - df['low']
    df['CombinedMomentum'] = df['AdjMomentum'] + df['GapOC'] + df['GapHL']
    
    # Confirm with ATR
    df['TrueRange'] = df[['high', 'low']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1)), abs(x['low'] - df['close'].shift(1))), axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    df['ATRAdjMomentum'] = df['CombinedMomentum'] / df['ATR']
    
    # Enhanced Reversal Sensitivity
    df['HighLowSpread'] = df['high'] - df['low']
    df['OpenCloseSpread'] = df['open'] - df['close']
    df['CombinedSpread'] = df['HighLowSpread'] - df['OpenCloseSpread']
    df['WeightedCombinedSpread'] = df['volume'] * df['CombinedSpread']
    
    # Measure Volume Impact
    df['5DayEMA_Volume'] = df['volume'].ewm(span=5, adjust=False).mean()
    df['VolumeRatio'] = df['volume'] / df['5DayEMA_Volume']
    df['VolAdjMomentum'] = df['ATRAdjMomentum'] * df['VolumeRatio']
    df['ReversalImpact'] = df['VolAdjMomentum'] * df['WeightedCombinedSpread']
    
    # Combine Components
    df['VolAdjComponent'] = df['ReversalImpact'] + df['VolAdjMomentum']
    df['FinalAlphaFactor'] = df['VolAdjComponent'] - df['WeightedCombinedSpread']
    
    return df['FinalAlphaFactor']
