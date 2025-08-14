import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Raw Momentum
    df['RawMomentum'] = df['close'] - df['close'].shift(20)
    
    # Adjust for Volume
    df['AvgVol'] = df['volume'].rolling(window=20).mean()
    df['VolRatio'] = df['volume'] / df['AvgVol']
    df['AdjMomentum'] = df['RawMomentum'] * df['VolRatio']
    
    # Incorporate Enhanced Price Sensitivity
    df['HLDiff'] = df['high'] - df['low']
    df['OCDiff'] = df['open'] - df['close']
    df['VolumeChange'] = df['volume'] - df['volume'].shift(1)
    df['HLWeight'] = df['VolumeChange'] * df['HLDiff']
    df['OCWeight'] = df['VolumeChange'] * df['OCDiff']
    df['CombinedSensitivity'] = df['HLWeight'] + df['OCWeight']
    
    # Incorporate Volume Trend Component
    df['EMAVolume'] = df['volume'].ewm(span=20, adjust=False).mean()
    df['VolumeTrend'] = df['volume'] - df['EMAVolume']
    
    # Incorporate Price Trend Component
    df['EMAClose'] = df['close'].ewm(span=20, adjust=False).mean()
    df['PriceTrend'] = df['close'] - df['EMAClose']
    
    # Confirm with ATR
    df['TrueRange'] = df[['high', 'low', df['close'].shift(1)]].max(axis=1) - df[['high', 'low', df['close'].shift(1)]].min(axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    
    # Final Factor
    df['CombinedMomentum'] = df['AdjMomentum'] + df['CombinedSensitivity']
    df['CombinedMomentum'] += df['VolumeTrend']
    df['CombinedMomentum'] += df['PriceTrend']
    df['FinalFactor'] = df['CombinedMomentum'] / df['ATR']
    
    return df['FinalFactor']
