import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Raw Momentum
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
    
    # Confirm with Volume Trend
    df['VolMA5'] = df['volume'].rolling(window=5).mean()
    df['VolMA20'] = df['volume'].rolling(window=20).mean()
    df['VolumeTrendAdjustment'] = (df['VolMA5'] > df['VolMA20']).astype(float) * 1.2 + (df['VolMA5'] <= df['VolMA20']).astype(float) * 0.8
    df['ConfirmedMomentum'] = df['CombinedMomentum'] * df['VolumeTrendAdjustment']
    
    # Adjust by ATR
    df['TrueRange'] = df[['high', 'low', 'close']].apply(lambda x: max(abs(x[0] - x[1]), abs(x[0] - x[2].shift(1)), abs(x[1] - x[2].shift(1))), axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    df['FinalFactor'] = df['ConfirmedMomentum'] / df['ATR']
    
    return df['FinalFactor']
