import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Simple Momentum
    df['SimpleMomentum'] = df['close'] - df['close'].shift(1)
    
    # Volume Adjusted Component
    df['DailyVolumeChange'] = df['volume'] - df['volume'].shift(1)
    df['VolumeAdjustedComponent'] = df['SimpleMomentum'] * df['DailyVolumeChange']
    
    # Price Reversal Sensitivity
    df['HighLowSpread'] = df['high'] - df['low']
    df['PriceReversalSensitivity'] = df['HighLowSpread'] * df['volume']
    
    # Intermediate Alpha Factor
    df['IntermediateAlphaFactor'] = df['VolumeAdjustedComponent'] - df['PriceReversalSensitivity']
    
    # Incorporate Enhanced Price Gaps
    df['GapOC'] = df['open'] - df['close']
    df['GapHL'] = df['high'] - df['low']
    df['CombinedMomentum'] = df['IntermediateAlphaFactor'] + df['GapOC'] + df['GapHL']
    
    # Adjust by ATR
    df['TrueRange'] = df[['high', 'low']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['close'].shift(1)), abs(x[1] - df['close'].shift(1))), axis=1)
    df['ATR_14'] = df['TrueRange'].rolling(window=14).mean()
    df['AdjMomentumATR'] = df['CombinedMomentum'] / df['ATR_14']
    
    # Final Adjustment
    df['VolMA_20'] = df['volume'].rolling(window=20).mean()
    df['FinalFactor'] = df['AdjMomentumATR'] * df['VolMA_20']
    
    return df['FinalFactor']
