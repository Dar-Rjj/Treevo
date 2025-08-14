import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Volatility
    intraday_volatility = (df['high'] - df['low']) / df['close']
    
    # Calculate Volume Percentage Change
    volume_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Combine Intraday Metrics
    combined_intraday_metrics = intraday_volatility * volume_change
    
    # Calculate Short-Term Momentum
    short_term_momentum = df['close'].rolling(window=5).mean() - df['close']
    
    # Calculate Long-Term Momentum
    long_term_momentum = df['close'].rolling(window=20).mean() - df['close']
    
    # Create a Momentum Differential
    momentum_differential = short_term_momentum - long_term_momentum
    
    # Calculate Volume Weighted Intraday Volatility
    vol_weighted_intraday_volatility = intraday_volatility * df['volume']
    
    # Combine Intraday Volatility and Volume Weighted Intraday Volatility
    combined_intraday_volatility = intraday_volatility + vol_weighted_intraday_volatility
    
    # Intraday Price Momentum
    high_low_diff = df['high'] - df['low']
    open_close_mom = df['close'] - df['open']
    intraday_price_momentum = 0.6 * high_low_diff + 0.4 * open_close_mom
    
    # Adjust for Volatility
    std_dev_5_days = df['close'].rolling(window=5).std()
    volatility_adjusted_momentum = intraday_price_momentum / std_dev_5_days
    
    # Adjust for Close-to-Open Reversal
    close_open_reversal = (df['open'] - df['close']) / df['close']
    
    # Integrate Intraday Volatility and Reversal Adjusted Momentum
    integrated_volatility_reversal = combined_intraday_volatility + close_open_reversal
    
    # Final Combined Factor
    final_factor = combined_intraday_metrics * momentum_differential + integrated_volatility_reversal
    
    # Smoothing
    final_factor = final_factor.ewm(span=10, adjust=False).mean()
    
    # Additional Momentum Confirmation
    ten_day_momentum = df['close'].rolling(window=10).mean() - df['close']
    if (ten_day_momentum > 0) & (short_term_momentum < 0):
        final_factor -= final_factor.abs() * 0.1
    elif (ten_day_momentum < 0) & (short_term_momentum > 0):
        final_factor += final_factor.abs() * 0.1
    
    # Introduce Volume Spike Detection
    volume_spike = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume'].rolling(window=20).mean()
    threshold = 2.0
    final_factor = np.where(volume_spike > threshold, final_factor * 1.1, final_factor)
    final_factor = np.where(volume_spike < -threshold, final_factor * 0.9, final_factor)
    
    return final_factor
