import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Simple Momentum
    df['SimpleMomentum'] = df['close'] / df['close'].shift(20) - 1
    
    # Volume Adjusted Component
    df['DailyVolumeChange'] = df['volume'] / df['volume'].shift(1)
    df['VolumeAdjustedMomentum'] = df['SimpleMomentum'] * df['DailyVolumeChange']
    
    # Enhanced Price Reversal Sensitivity
    df['HighLowSpread'] = df['high'] - df['low']
    df['OpenCloseSpread'] = abs(df['open'] - df['close'])
    df['WeightedSpreads'] = (df['HighLowSpread'] * df['DailyVolumeChange']) + (df['OpenCloseSpread'] * df['DailyVolumeChange'])
    
    # Volume Trend Component
    df['EMA_Volume'] = df['volume'].ewm(span=10, adjust=False).mean()
    df['VolumeTrend'] = df['volume'] / df['EMA_Volume'] - 1
    
    # Price Trend Component
    df['EMA_Close'] = df['close'].ewm(span=5, adjust=False).mean()
    df['PriceTrend'] = df['close'] / df['EMA_Close'] - 1
    
    # Incorporate Enhanced Price Gaps
    df['OpenCloseGap'] = df['open'] - df['close'].shift(1)
    df['HighLowGap'] = df['high'] - df['low']
    df['GapsMomentum'] = df['VolumeAdjustedMomentum'] + df['OpenCloseGap'] + df['HighLowGap']
    
    # Calculate Volume Surge
    avg_volume = df['volume'].rolling(window=10).mean()
    df['VolumeSurge'] = np.where(df['volume'] > 1.5 * avg_volume, 1, 0)
    
    # Measure Volume Impact
    df['VolumeImpact'] = df['volume'].ewm(span=10, adjust=False).mean()
    
    # Incorporate Volume Oscillations
    historical_avg_volume = df['volume'].rolling(window=10).mean()
    df['VolumeDifference'] = df['volume'] - historical_avg_volume
    df['VolumeOscillation'] = df['VolumeDifference'] / historical_avg_volume
    
    # Calculate Daily Gaps
    df['DailyGaps'] = df['open'] - df['close'].shift(1)
    
    # Calculate Volume Weighted Average of Gaps
    df['VolumeWeightedGaps'] = (df['DailyGaps'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Confirm with True Range
    df['TrueRange'] = df[['high', 'low', df['close'].shift(1)]].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), axis=1)
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    df['AdjustedAlpha'] = df['GapsMomentum'] / df['ATR']
    
    # Final Combination
    alpha_factor = (
        df['VolumeAdjustedMomentum'] -
        df['WeightedSpreads'] +
        df['VolumeTrend'] +
        df['PriceTrend'] +
        df['GapsMomentum'] * df['VolumeSurge'] -
        df['WeightedSpreads'] +
        df['VolumeOscillation'] +
        df['VolumeWeightedGaps'] +
        df['AdjustedAlpha']
    ) / df['VolumeImpact']
    
    return alpha_factor
