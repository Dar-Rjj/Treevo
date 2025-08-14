import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute Raw Momentum
    df['Close_t-20'] = df['close'].shift(20)
    df['RawMomentum'] = df['close'] - df['Close_t-20']
    
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
    df['ConfirmedMomentum'] = df['CombinedMomentum'] * (1.2 if df['VolMA5'] > df['VolMA20'] else 0.8)
    
    # Adjust Momentum by ATR
    df['TR1'] = df['high'] - df['low']
    df['TR2'] = (df['high'] - df['close'].shift(1)).abs()
    df['TR3'] = (df['low'] - df['close'].shift(1)).abs()
    df['TrueRange'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    df['AdjMomentumATR'] = df['ConfirmedMomentum'] / df['ATR']
    
    # Final Volume Adjustment
    df['VolMA_20'] = df['volume'].rolling(window=20).mean()
    df['FinalFactor'] = df['AdjMomentumATR'] * df['VolMA_20']
    
    # Volume-Price Adjusted Component
    df['DailyPriceVolumeEfficiency'] = (df['close'] - df['close'].shift(1)) / (df['volume'] + df['volume'].shift(1))
    df['FinalFactor'] = df['AdjMomentum'] * df['DailyPriceVolumeEfficiency']
    
    return df['FinalFactor']
