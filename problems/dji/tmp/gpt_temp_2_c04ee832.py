import pandas as pd
import pandas as pd

def heuristics_v2(df, n=5, m=7, spike_factor=2.0):
    # Calculate Intraday Range and Momentum
    df['intraday_range'] = df['high'] - df['low']
    df['close_momentum'] = df['close'] / df['close'].shift(n)
    
    # Measure Volume Activity Change
    df['volume_anomaly'] = df['volume'] - df['volume'].rolling(window=m).mean().shift(1)
    
    # Adjust Intraday Range by Volume Anomaly
    df['adjusted_intraday_range'] = df.apply(
        lambda row: row['intraday_range'] * (1 + row['volume_anomaly'] / df['volume']) if row['volume_anomaly'] > 0 else
                    row['intraday_range'] * (1 - abs(row['volume_anomaly'] / df['volume'])), 
        axis=1
    )
    
    # Incorporate Price Movement Intensity
    df['price_intensity'] = (df['close'] - df['open']) + (df['high'] - df['low'])
    
    # Generate Final Alpha Signal
    df['final_alpha_signal'] = df['adjusted_intraday_range'] * df['volume_anomaly'] * df['close_momentum'] * df['price_intensity']
    
    # Identify Volume Spikes
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['spike_threshold'] = df['volume_change'].rolling(window=m).median() * spike_factor
    df['spike_indicator'] = df['volume_change'] > df['spike_threshold']
    
    # Adjust Cumulative Moving Difference by Volume-Weighted Average
    df['cum_high_low_diff'] = df['high'] - df['low']
    df['vol_weighted_avg'] = df['volume'].rolling(window=m).sum() / m
    df['adj_cum_high_low_diff'] = df['cum_high_low_diff'] / df['vol_weighted_avg']
    
    # Adjust for Volume Spike
    df['spike_adjusted_diff'] = df['adj_cum_high_low_diff'] * df['spike_indicator'].astype(int)
    
    # Synthesize Final Alpha Factor
    df['final_alpha_factor'] = df['final_alpha_signal'] * (1 + df['spike_adjusted_diff'])
    
    return df['final_alpha_factor']
