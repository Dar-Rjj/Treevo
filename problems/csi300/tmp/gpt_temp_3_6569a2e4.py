import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Intraday Price Momentum
    df['intraday_momentum'] = df['close'] - df['open']
    
    # Detect Inside Day and Outside Day
    df['is_inside_day'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    df['is_outside_day'] = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
    
    # Volume-Weighted Close and Volume Imbalance
    df['volume_weighted_close'] = df['close'] * df['volume']
    df['volume_imbalance'] = df['volume'] - df['volume'].shift(1)
    
    # Bullish and Bearish Volume Spikes
    df['bullish_volume_spike'] = (df['volume'] > df['volume'].shift(1)) & (df['close'] > df['close'].shift(1))
    df['bearish_volume_spike'] = (df['volume'] > df['volume'].shift(1)) & (df['close'] < df['close'].shift(1))
    
    # Short-Term and Long-Term Moving Averages
    df['short_term_ma'] = df['close'].rolling(window=5).mean()
    df['long_term_ma'] = df['close'].rolling(window=20).mean()
    df['trend_strength'] = df['short_term_ma'] - df['long_term_ma']
    
    # Number of Consecutive Days with Positive and Negative Closes
    df['consecutive_positive_closes'] = (df['close'] > df['open']).astype(int).groupby((df['close'] < df['open']).cumsum()).cumsum()
    df['consecutive_negative_closes'] = (df['close'] < df['open']).astype(int).groupby((df['close'] > df['open']).cumsum()).cumsum()
    
    # True Range and Average True Range
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2].shift(1)), abs(x[1] - x[2].shift(1))), axis=1)
    df['average_true_range'] = df['true_range'].rolling(window=14).mean()
    
    # Upward and Downward Gaps
    df['upward_gap'] = (df['open'] > df['close'].shift(1)).astype(int)
    df['downward_gap'] = (df['open'] < df['close'].shift(1)).astype(int)
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (
        df['intraday_momentum'] +
        df['is_inside_day'].astype(int) * 1.0 +
        df['is_outside_day'].astype(int) * -1.0 +
        df['volume_weighted_close'] / df['volume'] +
        df['volume_imbalance'] * 0.5 +
        df['bullish_volume_spike'].astype(int) * 1.0 +
        df['bearish_volume_spike'].astype(int) * -1.0 +
        df['trend_strength'] * 0.5 +
        df['consecutive_positive_closes'] * 0.5 -
        df['consecutive_negative_closes'] * 0.5 +
        df['average_true_range'] * 0.5 +
        df['upward_gap'] * 1.0 -
        df['downward_gap'] * 1.0
    )
    
    return df['alpha_factor']
