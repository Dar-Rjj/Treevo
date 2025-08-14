import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 14-day Price Momentum
    df['PriceMomentum'] = df['Close'] - df['Close'].shift(14)
    
    # Identify Breakout Days
    df['HighLowRange'] = df['High'] - df['Low']
    avg_high_low_range = df['HighLowRange'].rolling(window=14).mean()
    df['Breakout'] = (df['HighLowRange'] > 2 * avg_high_low_range).astype(int)
    
    # Calculate Volume-Adjusted Breakout Impact
    df['DailyReturn'] = (df['Close'] - df['Open']) / df['Open']
    df['VolumeAdjustedReturn'] = df['DailyReturn'] * df['Volume'] * df['Breakout']
    df['VolumeAdjustedBreakoutImpact'] = df['VolumeAdjustedReturn'].rolling(window=14).sum()
    
    # Integrate Volume Trend Impact
    df['VolumeChange'] = df['Volume'] - df['Volume'].shift(1)
    weights = [0.2, 0.3, 0.5]
    df['WeightedVolumeTrend'] = df['VolumeChange'].rolling(window=3).apply(lambda x: (x * weights).sum(), raw=True)
    
    # Incorporate Volume Volatility
    df['VolumeVolatility'] = df['Volume'].rolling(window=14).std()
    
    # Combine All Factors
    df['CombinedFactor'] = (df['PriceMomentum'] + df['VolumeAdjustedBreakoutImpact']) * df['WeightedVolumeTrend'] / df['VolumeVolatility']
    
    return df['CombinedFactor']
