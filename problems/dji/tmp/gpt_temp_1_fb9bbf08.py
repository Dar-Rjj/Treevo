import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range and Momentum
    df['IntradayRange'] = df['high'] - df['low']
    df['Momentum'] = df['close'] / df['close'].shift(1)
    
    # Measure Volume Activity Change
    m = 5  # Example window size
    df['VolumeChange'] = df['volume'] - df['volume'].rolling(window=m).mean()
    
    # Combine Relative Strength, Volume Change, and Intraday Range
    df['RelativeStrength'] = (df['close'] / df['close'].shift(m))
    df['EnhancedIntradayRange'] = df['IntradayRange'] * (1 + df['VolumeChange'] / df['volume'].rolling(window=m).std())
    
    # Incorporate Price Movement Intensity
    df['HighLowRange'] = df['high'] - df['low']
    df['OpenCloseSpread'] = df['close'] - df['open']
    df['PriceMovementIntensity'] = df['HighLowRange'] + df['OpenCloseSpread']
    
    # Generate Final Alpha Signal
    df['AlphaSignal'] = df['EnhancedIntradayRange'] * df['VolumeChange'] * df['Momentum'] * df['PriceMovementIntensity']
    
    # Identify Volume Spikes
    df['VolumeSpikeThreshold'] = df['volume'].rolling(window=m).median() * 2
    df['VolumeSpikeIndicator'] = (df['VolumeChange'] > df['VolumeSpikeThreshold']).astype(int)
    
    # Adjust Cumulative Moving Difference by Volume-Weighted Average
    df['CumulativeMovingDifference'] = df['IntradayRange'].rolling(window=m).sum()
    df['VolumeWeightedAverage'] = df['volume'].rolling(window=m).sum() / m
    df['AdjustedCumulativeMovingDifference'] = df['CumulativeMovingDifference'] * df['VolumeWeightedAverage']
    
    # Adjust for Volume Spike
    df['AdjustedCumulativeMovingDifference'] *= df['VolumeSpikeIndicator']
    
    # Synthesize Final Alpha Factor
    df['SynthesizedAlphaFactor'] = df['AlphaSignal'] * df['AdjustedCumulativeMovingDifference']
    
    # Calculate Daily Price Momentum
    df['DailyPriceMomentum'] = df['close'] - df['close'].shift(1)
    
    # Calculate Short-Term Trend
    df['ShortTermTrend'] = df['DailyPriceMomentum'].ewm(span=5).mean() / df['DailyPriceMomentum'].rolling(window=5).std()
    
    # Calculate Long-Term Trend
    df['LongTermTrend'] = df['DailyPriceMomentum'].ewm(span=20).mean() / df['DailyPriceMomentum'].rolling(window=20).std()
    
    # Generate Volume Synchronized Oscillator
    df['VolumeSynchronizedOscillator'] = (df['LongTermTrend'] - df['ShortTermTrend']) * df['volume']
    
    # Adjust Momentum by Inverse of Volatility
    df['TrueRange'] = df[['high' - 'low', 'high' - 'close'].shift(1), 'close'].shift(1) - 'low']].max(axis=1)
    df['AdjustedMomentum'] = df['Momentum'] / df['TrueRange']
    
    # Integrate Combined Factors
    df['IntegratedFactor'] = df['SynthesizedAlphaFactor'] * df['AdjustedIntradayRange'] + df['OpenCloseSpread'] * df['AdjustedMomentum'] * df['VolumeSynchronizedOscillator']
    
    return df['IntegratedFactor']
