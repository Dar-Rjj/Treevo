import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Simple Momentum
    df['SimpleMomentum'] = df['close'] / df['close'].shift(20) - 1
    
    # Adjust for Volume
    avg_vol = df['volume'].rolling(window=20).mean()
    vol_ratio = df['volume'] / avg_vol
    df['AdjMomentum'] = df['SimpleMomentum'] * vol_ratio
    
    # Incorporate Enhanced Price Gaps
    df['GapOC'] = df['open'] - df['close']
    df['GapHL'] = df['high'] - df['low']
    df['CombinedMomentum'] = df['AdjMomentum'] + df['GapOC'] + df['GapHL']
    
    # Confirm with Volume
    vol_ma5 = df['volume'].rolling(window=5).mean()
    vol_ma20 = df['volume'].rolling(window=20).mean()
    df['ConfirmedMomentum'] = df['CombinedMomentum'] * (1.2 if vol_ma5 > vol_ma20 else 0.8)
    
    # Adjust Momentum by ATR
    df['TrueRange'] = df.apply(lambda row: max(row['high'] - row['low'], 
                                                abs(row['high'] - df['close'].shift(1)), 
                                                abs(row['low'] - df['close'].shift(1))), axis=1)
    atr = df['TrueRange'].rolling(window=14).mean()
    df['AdjMomentumATR'] = df['ConfirmedMomentum'] / atr
    
    # Measure Volume Impact
    vol_ema5 = df['volume'].ewm(span=5, adjust=False).mean()
    volume_ratio = df['volume'] / vol_ema5
    df['VolumeAdjustedFactor'] = df['AdjMomentumATR'] * volume_ratio
    
    # Calculate Close-to-High and Close-to-Low Distances
    df['CloseToHigh'] = df['high'] - df['close']
    df['CloseToLow'] = df['close'] - df['low']
    
    # Combine Factors with Close-to-Low and Close-to-High Distances
    df['CombinedFactor1'] = df['VolumeAdjustedFactor'] * df['CloseToLow']
    df['CombinedFactor2'] = df['VolumeAdjustedFactor'] * df['CloseToHigh']
    
    # Enhanced Price Reversal Sensitivity
    df['HCSpread'] = df['high'] - df['close']
    df['OLSpread'] = df['open'] - df['low']
    df['WeightedHC'] = df['HCSpread'] * df['volume']
    df['WeightedOL'] = df['OLSpread'] * df['volume']
    df['TotalWeightedSpreads'] = df['WeightedHC'] + df['WeightedOL']
    
    # Final Alpha Factor
    df['FinalCombinedFactor'] = df['CombinedFactor1'] + df['CombinedFactor2']
    df['FinalAlpha'] = df['FinalCombinedFactor'] - df['TotalWeightedSpreads']
    
    return df['FinalAlpha']
