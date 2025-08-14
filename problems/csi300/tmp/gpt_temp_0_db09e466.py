import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Movement
    high_low_range = df['high'] - df['low']
    close_open_diff = df['close'] - df['open']
    
    # Incorporate Volume Influence
    volume_adjusted_momentum = (df['volume'] * (high_low_range + close_open_diff))
    
    # Adaptive Smoothing via Moving Average
    ema_period = 10  # Placeholder for dynamic EMA period calculation
    smoothed_vol_adj_mom = volume_adjusted_momentum.ewm(span=ema_period, adjust=False).mean()
    
    # Adjust for Market Volatility
    daily_return = df['close'].pct_change()
    abs_daily_return = np.abs(daily_return)
    robust_market_volatility = abs_daily_return.rolling(window=30).apply(lambda x: np.median(np.abs(x - np.median(x))))
    modified_vol_adj_mom = smoothed_vol_adj_mom / robust_market_volatility
    
    # Incorporate Trend Reversal Signal
    short_term_momentum = df['close'].ewm(span=5, adjust=False).mean()
    long_term_momentum = df['close'].ewm(span=20, adjust=False).mean()
    momentum_reversal = short_term_momentum - long_term_momentum
    reversal_points = np.sign(momentum_reversal)  # Identify reversal points
    
    # Integrate Non-Linear Transformation
    sqrt_transformed_mom = np.sqrt(modified_vol_adj_mom)
    log_transformed_mom = np.log(modified_vol_adj_mom)
    
    # Enhance Reversal Signal with Adaptive Smoothing
    smoothed_reversal_signal = reversal_points.ewm(span=ema_period, adjust=False).mean()
    combined_signal = smoothed_reversal_signal + sqrt_transformed_mom + log_transformed_mom
    
    # Refine Final Alpha Factor
    final_alpha_factor = combined_signal.ewm(span=ema_period, adjust=False).mean()
    
    return final_alpha_factor
