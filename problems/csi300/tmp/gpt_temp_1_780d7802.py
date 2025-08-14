import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Raw Momentum
    df['Close_t'] = df['close']
    df['Close_t-20'] = df['close'].shift(20)
    df['RawMomentum'] = df['Close_t'] - df['Close_t-20']

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
    df['ConfirmedMomentum'] = df['CombinedMomentum'] * (df['VolMA5'] > df['VolMA20']).astype(int) * 1.0 + df['CombinedMomentum'] * (df['VolMA5'] <= df['VolMA20']).astype(int) * 0.5

    # Adjust by ATR and Final Volume
    df['TR'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2].shift(1)), abs(x[1] - x[2].shift(1))), axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['AdjMomentumATR'] = df['ConfirmedMomentum'] / df['ATR']
    df['VolMA_20'] = df['volume'].rolling(window=20).mean()
    df['FinalFactor'] = df['AdjMomentumATR'] * df['VolMA_20']

    return df['FinalFactor'].dropna()
