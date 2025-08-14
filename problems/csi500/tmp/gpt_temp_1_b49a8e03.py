import pandas as pd
import pandas as pd

def heuristics_v2(df, N=20, spike_factor=2, trend_factor_pos=1.2, trend_factor_neg=0.8, volatility_threshold=0.05, volatility_factor=1.5):
    # Calculate Daily Return
    df['DailyReturn'] = df['close'].pct_change()
    
    # Identify Volume Spike
    df['VolumeChange'] = df['volume'].diff()
    df['Spike'] = (df['VolumeChange'] / df['volume'].shift(1)) > spike_factor
    
    # Calculate Price Volatility
    df['DailyHighLowRange'] = df['high'] - df['low']
    df['NDayAverageHighLowRange'] = df['DailyHighLowRange'].rolling(window=N).mean()
    
    # Calculate Volume-Weighted N-day Momentum
    df['VolumeWeightedMomentum'] = (df['DailyReturn'] * df['volume']).rolling(window=N).sum()
    df['AdjustedVolumeWeightedMomentum'] = df['VolumeWeightedMomentum'] * (spike_factor if df['Spike'] else 1)
    
    # Calculate Price Trend
    df['NDayEMA'] = df['close'].ewm(span=N, adjust=False).mean()
    df['Trend'] = df['close'] > df['NDayEMA']
    
    # Final Alpha Factor
    df['TrendAdjustedMomentum'] = df['AdjustedVolumeWeightedMomentum'] * (trend_factor_pos if df['Trend'] else trend_factor_neg)
    df['AlphaFactor'] = df['TrendAdjustedMomentum'] * (volatility_factor if df['NDayAverageHighLowRange'] > volatility_threshold else 1)
    
    return df['AlphaFactor']

# Example usage:
# alpha_factor = heuristics_v2(df)
