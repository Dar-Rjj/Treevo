import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

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
    df['VolWMA5'] = df['volume'].rolling(window=5).apply(lambda x: np.dot(x, np.arange(1, 6)) / 15.0, raw=True)
    df['VolWMA20'] = df['volume'].rolling(window=20).apply(lambda x: np.dot(x, np.arange(1, 21)) / 210.0, raw=True)
    df['ConfirmedMomentum'] = df['CombinedMomentum'].where(df['VolWMA5'] > df['VolWMA20'], df['CombinedMomentum'] * 0.8) * 1.2
    
    # Adjust Momentum by ATR
    df['PrevClose'] = df['close'].shift(1)
    df['TrueRange'] = df[['high', 'low', 'PrevClose']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    df['AdjMomentumATR'] = df['ConfirmedMomentum'] / df['ATR']
    
    # Introduce Final Volume Adjustment
    df['VolWMA_10'] = df['volume'].rolling(window=10).apply(lambda x: np.dot(x, np.arange(1, 11)) / 55.0, raw=True)
    df['FinalFactor'] = df['AdjMomentumATR'] * df['VolWMA_10']
    
    return df['FinalFactor'].dropna()

# Example usage:
# df = pd.DataFrame({
#     'date': pd.date_range(start='2023-01-01', periods=50),
#     'open': np.random.rand(50) * 100,
#     'high': np.random.rand(50) * 100,
#     'low': np.random.rand(50) * 100,
#     'close': np.random.rand(50) * 100,
#     'amount': np.random.rand(50) * 1000,
#     'volume': np.random.randint(1000, 10000, size=50)
# })
# df.set_index('date', inplace=True)
# factor = heuristics_v2(df)
