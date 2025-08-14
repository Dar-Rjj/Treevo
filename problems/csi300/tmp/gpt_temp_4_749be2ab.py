import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Simple Momentum
    n = 10  # Example: 10-day momentum
    df['SimpleMomentum'] = df['close'].pct_change(n)
    
    # Volume Adjusted Component
    df['VolumeChange'] = df['volume'].pct_change()
    df['VolumeAdjustedMomentum'] = df['SimpleMomentum'] * df['VolumeChange']
    
    # Price Reversal Sensitivity
    df['HighLowSpread'] = df['high'] - df['low']
    df['WeightedReversal'] = df['HighLowSpread'] * (df['volume'] / df['volume'].rolling(window=5).mean())
    
    # Combine Components
    df['IntermediateAlphaFactor'] = df['VolumeAdjustedMomentum'] - df['WeightedReversal']
    
    # Incorporate Enhanced Price Gaps
    df['GapOC'] = df['open'] - df['close'].shift(1)
    df['GapHL'] = df['high'] - df['low']
    df['CombinedMomentum'] = df['IntermediateAlphaFactor'] + df['GapOC'] + df['GapHL']
    
    # Confirm with Volume
    df['VolMA5'] = df['volume'].rolling(window=5).mean()
    df['VolMA20'] = df['volume'].rolling(window=20).mean()
    df['ConfirmedMomentum'] = df['CombinedMomentum'] * (df['VolMA5'] > df['VolMA20']).astype(int) + df['CombinedMomentum'] * 0.5 * (df['VolMA5'] <= df['VolMA20']).astype(int)
    
    # Adjust by ATR
    df['TrueRange'] = df[['high', 'low']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['close'].shift(1)), abs(x[1] - df['close'].shift(1))), axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    df['AdjMomentumATR'] = df['ConfirmedMomentum'] / df['ATR']
    
    # Final Adjustment
    df['VolMA_20'] = df['volume'].rolling(window=20).mean()
    df['FinalFactor'] = df['AdjMomentumATR'] * df['VolMA_20']
    
    return df['FinalFactor']
