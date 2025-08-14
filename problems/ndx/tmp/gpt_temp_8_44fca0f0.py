import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-to-Low Range
    high_low_range = (df['high'] - df['low']) / df['low']
    
    # Integrate Intraday and Multi-Day Momentum with Volume Adjusted Momentum
    intraday_momentum = (df['high'] - df['low']) / df['low'] * df['volume'] / (df['high'] + df['low'])
    sequential_5day_momentum = (df['close'].pct_change().rolling(5).sum() / 5)
    daily_log_returns = np.log(df['close'] / df['close'].shift(1))
    ema_20_log_returns = daily_log_returns.ewm(span=20, adjust=False).mean()
    volume_weighted_momentum = (intraday_momentum * df['volume'].rolling(5).mean()) + \
                               (sequential_5day_momentum * df['volume']) + \
                               (ema_20_log_returns * df['volume'])
    
    # Calculate Daily Price Change
    daily_price_change = df['close'] - df['open']
    
    # Compute 5-Day and 10-Day EMAs of Price Change
    ema_5_price_change = daily_price_change.ewm(span=5, adjust=False).mean()
    ema_10_price_change = daily_price_change.ewm(span=10, adjust=False).mean()
    
    # Determine Reversal Signal
    reversal_signal = (ema_5_price_change > ema_10_price_change) - (ema_5_price_change < ema_10_price_change)
    rolling_median_high_low_range = high_low_range.rolling(14).median()
    amplified_reversal_signal = reversal_signal * (rolling_median_high_low_range > rolling_median_high_low_range.median())
    
    # Filter by Volume
    volume_threshold = df['volume'].quantile(0.75)
    filtered_signal = amplified_reversal_signal.where(df['volume'] > volume_threshold, 0)
    
    # Incorporate Enhanced Volatility into Alpha Factor
    historical_volatility = df['close'].rolling(21).std()
    moving_sd_high_low = high_low_range.rolling(20).std()
    open_to_prev_close_diff = (df['open'] - df['close'].shift(1)).abs()
    moving_sd_open_to_prev_close = open_to_prev_close_diff.rolling(20).std()
    volatility_adjusted_factor = 1 / (historical_volatility * moving_sd_high_low * moving_sd_open_to_prev_close)
    
    # Final Alpha Factor
    alpha_factor = (volume_weighted_momentum + 
                    ema_20_log_returns * df['volume'] + 
                    filtered_signal +
                    volatility_adjusted_factor)
    
    return alpha_factor
