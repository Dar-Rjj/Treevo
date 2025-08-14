import numpy as np
def heuristics_v2(df):
    # Daily Range Factor
    df['daily_range'] = df['high'] - df['low']
    
    # Relative High-Low to Close Factor
    df['rel_high_close'] = (df['high'] - df['open']) / df['close']
    df['rel_low_close'] = (df['open'] - df['low']) / df['close']
    
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'].diff()
    
    # Compute Weighted Moving Averages (WMA)
    def weighted_moving_average(series, weights):
        return np.convolve(series, weights, 'valid') / sum(weights)
