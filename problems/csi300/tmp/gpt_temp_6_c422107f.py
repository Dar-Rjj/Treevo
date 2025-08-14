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
    df['ConfirmedMomentum'] = df['CombinedMomentum'] * (df['VolMA5'] > df['VolMA20']).astype(float) * 1.0 + \
                              df['CombinedMomentum'] * (df['VolMA5'] <= df['VolMA20']).astype(float) * 0.5
    
    # Adjust Momentum by ATR
    df['TR'] = df[['high' - 'low', 'high' - df['close'].shift(1), 'low' - df['close'].shift(1)]].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['AdjMomentumATR'] = df['ConfirmedMomentum'] / df['ATR']
    
    # Introduce Final Volume Adjustment
    df['VolMA_20'] = df['volume'].rolling(window=20).mean()
    df['FinalFactor'] = df['AdjMomentumATR'] * df['VolMA_20']
    
    # Volume-Price Adjusted Component
    df['DailyPriceVolumeEfficiency'] = (df['close'] - df['close'].shift(1)) / (df['volume'] + df['volume'].shift(1))
    df['VPAdjComponent'] = df['AdjMomentum'] * df['DailyPriceVolumeEfficiency']
    
    # Enhanced Volume Adjusted Momentum
    df['DailyVolumeChange'] = df['volume'] - df['volume'].shift(1)
    df['EVAMComponent'] = df['AdjMomentum'] * df['DailyVolumeChange']
    
    # Enhanced Price Reversal Sensitivity
    df['HighLowRangeToCloseRatio'] = (df['high'] - df['low']) / df['close']
    df['WeightedHighLowRange'] = df['volume'] * df['HighLowRangeToCloseRatio']
    df['OpenCloseSpread'] = df['open'] - df['close']
    df['WeightedOpenCloseSpread'] = df['volume'] * df['OpenCloseSpread']
    df['CombinedWeightedSpreads'] = df['WeightedHighLowRange'] + df['WeightedOpenCloseSpread']
    
    # Calculate Daily Gaps
    df['DailyGap'] = df['open'] - df['close'].shift(1)
    
    # Volume-Adjusted Gap
    df['VolumeAdjustedGap'] = (df['DailyGap'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Detect Volume Spikes
    df['VolumeSpike'] = df['volume'] - df['volume'].shift(1)
    
    # Final Alpha Factor
    df['AlphaFactor'] = df['FinalFactor'] + df['VPAdjComponent'] + df['EVAMComponent'] - df['CombinedWeightedSpreads'] + df['VolumeAdjustedGap'] + df['VolumeSpike']
    
    return df['AlphaFactor']
