import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['low']
    
    # Weight by Average Volume
    df['average_volume'] = df['volume'] / 2
    df['intraday_momentum'] = df['intraday_return'] * df['average_volume']
    
    # Calculate Volume Ratio
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Adjust for Intraday Volatility
    df['intraday_range'] = df['high'] - df['low']
    df['normalized_intraday_range'] = df['intraday_range'] / df['close']
    df['volume_thrust'] = df['volume_ratio'] / (1 + df['normalized_intraday_range'])
    
    # Combine Intraday Momentum and Volume Thrust
    df['indicator'] = df['intraday_momentum'] * df['volume_thrust']
    
    # Smooth the Indicator
    window_size = 5
    df['smoothed_indicator'] = df['indicator'].rolling(window=window_size).mean()
    
    # Adjust for Recent Trend
    trend_window = 3
    df['recent_trend_slope'] = df['smoothed_indicator'].rolling(window=trend_window).apply(lambda x: np.polyfit(range(trend_window), x, 1)[0], raw=False)
    df['final_alpha_factor'] = df['smoothed_indicator'] + df['recent_trend_slope']
    
    return df['final_alpha_factor']
