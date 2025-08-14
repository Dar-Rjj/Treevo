import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 10-day Average Price Range
    df['PriceRange'] = df['high'] - df['low']
    df['10DayAvgPriceRange'] = df['PriceRange'].rolling(window=10).mean()
    
    # Compute Volume-Adjusted Momentum
    df['Momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['VolumeAdjustedMomentum'] = df['Momentum'] * df['volume']
    
    # Calculate Intraday Momentum
    df['IntradayRange'] = df['high'] - df['low']
    
    # Calculate Volume Spike
    df['VolumeSpike'] = (df['volume'] > 1.5 * df['volume'].shift(1)).astype(int)
    
    # Adjust Intraday Momentum for Volume Spike
    df['AdjustedIntradayMomentum'] = df['IntradayRange'] * (2 if df['VolumeSpike'] == 1 else 1)
    
    # Combine Metrics
    df['CombinedMetric'] = (df['VolumeAdjustedMomentum'] / df['10DayAvgPriceRange']) + df['AdjustedIntradayMomentum']
    
    # Weight by Volume
    df['WeightedCombinedMetric'] = df['CombinedMetric'] * df['volume']
    
    # Calculate Short-Term and Long-Term Exponential Moving Averages of Weighed Combined Value
    df['ShortTermEMA'] = df['WeightedCombinedMetric'].ewm(span=12, adjust=False).mean()
    df['LongTermEMA'] = df['WeightedCombinedMetric'].ewm(span=26, adjust=False).mean()
    
    # Calculate Divergence
    df['Divergence'] = df['LongTermEMA'] - df['ShortTermEMA']
    
    # Apply Sign Function
    df['AlphaFactor'] = df['Divergence'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    return df['AlphaFactor']
