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
    df['ConfirmedMomentum'] = df['CombinedMomentum'].where(df['VolMA5'] > df['VolMA20'], df['CombinedMomentum'] * 0.8) * 1.2
    
    # Adjust by ATR
    df['TR'] = df[['high', 'low']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['close'].shift(1)), abs(x[1] - df['close'].shift(1))), axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['FinalFactor'] = df['ConfirmedMomentum'] / df['ATR']
    
    # Include Price Volatility
    df['StdDev10'] = df['close'].rolling(window=10).std()
    df['AdjustedFinalFactor'] = df['FinalFactor'] * (1 + df['StdDev10'] / 100)
    
    # Apply Price Breakout Filter
    df['High50'] = df['high'].rolling(window=50).max()
    df['Low50'] = df['low'].rolling(window=50).min()
    df['PriceRange'] = (df['high'] - df['low']) / (df['High50'] - df['Low50'])
    df['AdjustedFinalFactor'] = df['AdjustedFinalFactor'].where(df['PriceRange'] > 0.5, df['AdjustedFinalFactor'] * 0.9) * 1.1
    
    return df['AdjustedFinalFactor']
