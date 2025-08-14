import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Momentum Differential
    short_term_mom = df['close'] - df['close'].rolling(window=5).mean()
    long_term_mom = df['close'] - df['close'].rolling(window=20).mean()
    momentum_diff = long_term_mom - short_term_mom
    
    # Intraday Price Momentum
    high_low_diff = df['high'] - df['low']
    open_close_diff = df['close'] - df['open']
    intraday_momentum = (high_low_diff + open_close_diff) / 2
    close_std_dev = df['close'].rolling(window=10).std()
    intraday_momentum_adj = intraday_momentum / close_std_dev
    
    # Volume Weighting and Adjustment
    volume_mom = df['volume'] * momentum_diff
    volume_increase = df['volume'] > df['volume'].shift(1)
    volume_weighted_mom = volume_mom * (1.5 if volume_increase else 1.0)
    
    # Volatility Adjustment
    close_10d_std = df['close'].rolling(window=10).std()
    volatility_adjusted_mom = volume_weighted_mom / close_10d_std
    
    # Trend Following
    ma_50 = df['close'].rolling(window=50).mean()
    trend_factor = np.where(df['close'] > ma_50, 1.2, 0.8)
    trend_adjusted_mom = volatility_adjusted_mom * trend_factor
    
    # Integrate and Smooth
    final_momentum = trend_adjusted_mom + intraday_momentum_adj
    final_smoothed_momentum = final_momentum.ewm(span=10).mean()
    
    return final_smoothed_momentum
