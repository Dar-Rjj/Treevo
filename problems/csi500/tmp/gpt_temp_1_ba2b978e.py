import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Identify Volume Spike
    df['volume_percentage_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    threshold = 0.5  # Define a threshold for volume spike
    df['volume_spike'] = df['volume_percentage_change'] > threshold
    
    # Adjust High-Low Range by Volume Spike
    volume_spike_factor = 1.5  # Define a factor to amplify the high-low range when there's a volume spike
    df['adjusted_high_low_range'] = df['high_low_range'] * (volume_spike_factor if df['volume_spike'] else 1)
    
    # Incorporate Price Trend
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['price_trend'] = 0
    df.loc[df['close'] > df['sma_5'], 'price_trend'] = 1
    df.loc[df['close'] < df['sma_10'], 'price_trend'] = -1
    
    # Final Alpha Factor
    df['alpha_factor'] = df['adjusted_high_low_range'] * df['price_trend']
    df['alpha_factor'] = df['alpha_factor'].apply(lambda x: max(0, x))  # Ensure positive values
    
    return df['alpha_factor']
