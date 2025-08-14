import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Short-Term and Long-Term Momentum
    short_term_momentum = df['close'].rolling(window=5).mean()
    long_term_momentum = df['close'].rolling(window=20).mean()
    
    # Momentum Differential
    momentum_differential = long_term_momentum - short_term_momentum
    
    # Intraday Momentum Components
    high_low_difference = df['high'] - df['low']
    open_close_momentum = df['close'] - df['open']
    
    # Combine Intraday Momentum Components
    combined_intraday_momentum = (high_low_difference + open_close_momentum) / 2
    volatility_adjusted_momentum = combined_intraday_momentum / df['close'].rolling(window=20).std()
    
    # Volume-Weighted Intraday Momentum
    volume_weighted_momentum = volatility_adjusted_momentum * df['volume']
    
    # Final Integrated Momentum Differential
    integrated_momentum = momentum_differential + volume_weighted_momentum
    
    # Volume Confirmation
    volume_change = df['volume'].pct_change()
    integrated_momentum_boosted = np.where(volume_change > 0, integrated_momentum * 1.1, integrated_momentum)
    
    # Smooth with Exponential Moving Average
    smoothed_momentum = integrated_momentum_boosted.ewm(span=10).mean()
    
    # Incorporate 25-Day Momentum
    momentum_25_day = df['close'].pct_change(periods=25)
    final_momentum = smoothed_momentum + momentum_25_day
    
    # Adjust for Close-to-Open Reversal
    close_to_open_reversal = (df['open'] - df['close']) / df['close']
    final_factor = final_momentum * close_to_open_reversal
    
    return final_factor
