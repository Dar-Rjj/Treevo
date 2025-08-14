import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Price Momentum
    df['MA14'] = df['close'].rolling(window=14).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MomentumScore'] = (df['MA14'] > df['MA50']).astype(int) * 2 - 1  # Positive if MA14 > MA50, else negative
    
    # Volume Increase on Price Rise
    df['VolumeIncrease'] = ((df['close'] > df['close'].shift(1)) & 
                            (df['volume'] > df['volume'].shift(1))).astype(int)
    df['VolumeChangePct'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['VolumeScore'] = (df['VolumeChangePct'] > 0.1).astype(int) * df['VolumeIncrease']
    
    # Price Volatility
    df['DailyRange'] = df['high'] - df['low']
    df['AvgDailyRange'] = df['DailyRange'].rolling(window=20).mean()
    df['VolatilityFactor'] = (df['DailyRange'] > df['AvgDailyRange']).astype(int)
    
    # Money Flow Index (MFI) Based Factor
    df['TypicalPrice'] = (df['high'] + df['low'] + df['close']) / 3
    df['MoneyFlow'] = df['TypicalPrice'] * df['volume']
    df['PositiveMF'] = df.apply(lambda x: x['MoneyFlow'] if x['TypicalPrice'] > df['TypicalPrice'].shift(1) else 0, axis=1)
    df['NegativeMF'] = df.apply(lambda x: x['MoneyFlow'] if x['TypicalPrice'] < df['TypicalPrice'].shift(1) else 0, axis=1)
    df['PositiveMF14'] = df['PositiveMF'].rolling(window=14).sum()
    df['NegativeMF14'] = df['NegativeMF'].rolling(window=14).sum()
    df['MFI'] = 100 - (100 / (1 + (df['PositiveMF14'] / df['NegativeMF14'].abs())))
    df['MFIScore'] = ((df['MFI'] < 20) * 2 - 1) - ((df['MFI'] > 80) * 2 - 1)
    
    # Combine all factors into a single alpha factor
    df['AlphaFactor'] = df['MomentumScore'] + df['VolumeScore'] + df['VolatilityFactor'] + df['MFIScore']
    
    return df['AlphaFactor']
