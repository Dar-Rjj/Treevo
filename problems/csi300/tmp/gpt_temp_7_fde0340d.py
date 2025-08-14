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
    
    # Confirm with Volume
    df['VolMA5'] = df['volume'].rolling(window=5).mean()
    df['VolMA20'] = df['volume'].rolling(window=20).mean()
    df['ConfirmedMomentum'] = df['CombinedMomentum'].where(df['VolMA5'] > df['VolMA20'], 0.8 * df['CombinedMomentum'])
    df['ConfirmedMomentum'] = df['ConfirmedMomentum'].where(df['VolMA5'] <= df['VolMA20'], 1.2 * df['ConfirmedMomentum'])
    
    # Adjust Momentum by ATR
    df['TR'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2].shift(1)), abs(x[1] - x[2].shift(1))), axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['AdjMomentumATR'] = df['ConfirmedMomentum'] / df['ATR']
    
    # Introduce Final Volume Adjustment
    df['VolMA_20'] = df['volume'].rolling(window=20).mean()
    df['FinalFactor'] = df['AdjMomentumATR'] * df['VolMA_20']
    
    # Volume-Price Adjusted Component with Reversal Sensitivity
    df['DailyPriceVolumeEfficiency'] = (df['close'] - df['close'].shift(1)) / (df['volume'] + df['volume'].shift(1))
    df['WeightedMomentum'] = df['AdjMomentum'] * df['DailyPriceVolumeEfficiency']
    
    # Combine Weighted Spreads with Multi-Level Reversal Sensitivity
    df['HighLowSpread'] = df['high'] - df['low']
    df['OpenCloseSpread'] = df['open'] - df['close']
    df['VolumeChange'] = df['volume'] / df['volume'].shift(1)
    df['WeightedHighLowSpread'] = df['HighLowSpread'] * df['VolumeChange']
    df['WeightedOpenCloseSpread'] = df['OpenCloseSpread'] * df['VolumeChange']
    df['CombinedSpreads'] = df['WeightedHighLowSpread'] + df['WeightedOpenCloseSpread']
    df['ReversalSensitivity'] = df['CombinedSpreads'].rolling(window=5).mean()
    
    # Add Volume Trend Component
    df['EMA_Volume'] = df['volume'].ewm(span=20, adjust=False).mean()
    df['VolumeTrend'] = df['volume'] / df['EMA_Volume']
    
    # Detect Significant Volume Spikes
    df['PrevVolume'] = df['volume'].shift(1).rolling(window=5).mean()
    df['VolumeSpike'] = df['volume'] - df['PrevVolume']
    df['FinalFactor'] += df['VolumeSpike']
    
    # Measure Volume Impact
    df['EMA15_Volume'] = df['volume'].ewm(span=15, adjust=False).mean()
    df['FinalFactor'] /= df['EMA15_Volume']
    
    return df['FinalFactor']
