import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Movement
    high_low_range = df['high'] - df['low']
    close_open_diff = df['close'] - df['open']
    
    # Incorporate Volume Influence
    volume_adjusted_momentum = (high_low_range + close_open_diff) * df['volume']
    
    # Adaptive Smoothing via Moving Average
    def dynamic_ema(data, span):
        return data.ewm(span=span).mean()
    
    recent_volatility = df['close'].pct_change().rolling(window=30).std()
    ema_period = 10 + (recent_volatility * 10).astype(int)
    smoothed_volume_adjusted_momentum = dynamic_ema(volume_adjusted_momentum, span=ema_period)
    
    # Adjust for Market Volatility
    daily_return = df['close'].pct_change()
    abs_daily_return = np.abs(daily_return)
    robust_market_volatility = abs_daily_return.rolling(window=30).median() * 1.4826  # MAD to SD conversion
    modified_volume_adjusted_momentum = smoothed_volume_adjusted_momentum / robust_market_volatility
    
    # Incorporate Trend Reversal Signal
    short_term_momentum = df['close'].ewm(span=5).mean()
    long_term_momentum = df['close'].ewm(span=20).mean()
    momentum_reversal = short_term_momentum - long_term_momentum
    reversal_points = np.sign(momentum_reversal.diff())
    
    # Integrate Non-Linear Transformation
    sqrt_transformed_momentum = np.sqrt(modified_volume_adjusted_momentum)
    log_transformed_momentum = np.log(modified_volume_adjusted_momentum)
    non_linear_transformed_momentum = (sqrt_transformed_momentum + log_transformed_momentum) / 2
    
    # Enhance Reversal Signal with Adaptive Smoothing
    smoothed_reversal_signal = dynamic_ema(reversal_points, span=ema_period)
    interim_alpha_factor = non_linear_transformed_momentum + smoothed_reversal_signal
    
    # Final Adaptive Smoothing
    final_alpha_factor = dynamic_ema(interim_alpha_factor, span=ema_period)
    
    return final_alpha_factor
