import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Price Returns
    df['Close'] = df['close']
    df['NextDayOpen'] = df['open'].shift(-1)
    df['SimpleReturns'] = (df['NextDayOpen'] - df['Close']) / df['Close']
    df['VolumeWeightedReturns'] = df['SimpleReturns'] * df['volume']
    
    # Identify Volume Surge Days
    df['VolumeChange'] = df['volume'] - df['volume'].shift(1)
    df['RollingMeanVolume'] = df['volume'].rolling(window=5).mean()
    df['IsVolumeSurge'] = df['volume'] > df['RollingMeanVolume']
    
    # Calculate Volatility
    df['HighLowRange'] = df['high'] - df['low']
    df['HighCloseRange'] = (df['high'] - df['close'].shift(1)).abs()
    df['LowCloseRange'] = (df['low'] - df['close'].shift(1)).abs()
    df['TrueRange'] = df[['HighLowRange', 'HighCloseRange', 'LowCloseRange']].max(axis=1)
    lookback_period = 5
    df['ATR'] = df['TrueRange'].rolling(window=lookback_period).mean()
    
    # Adjust Volume-Weighted Returns by Volatility
    df['AdjustedReturns'] = df['VolumeWeightedReturns'] / df['ATR']
    
    # Combine Adjusted Returns with Volume Surge Indicator
    surge_factor = 1.5
    df['FinalFactor'] = df['AdjustedReturns'] * (surge_factor if df['IsVolumeSurge'] else 1)
    
    return df['FinalFactor']
