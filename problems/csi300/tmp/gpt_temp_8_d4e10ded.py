import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Enhanced Intraday Volatility
    intraday_volatility = (df['high'] - df['low']) / df['close']
    
    # Calculate Intraday Volume Change
    volume_change = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Combine Intraday Metrics
    intraday_metrics = intraday_volatility * volume_change
    
    # Compute Intraday Momentum Intensity
    high_low_ratio = df['high'] / df['low']
    open_close_diff = df['close'] - df['open']
    intraday_momentum_intensity = (high_low_ratio + open_close_diff) * df['volume']
    
    # Calculate Short-Term Momentum
    short_term_avg = df['close'].rolling(window=5).mean()
    short_term_momentum = short_term_avg - df['close']
    
    # Calculate Long-Term Momentum
    long_term_avg = df['close'].rolling(window=20).mean()
    long_term_momentum = long_term_avg - df['close']
    
    # Create a Momentum Differential
    momentum_differential = long_term_momentum - short_term_momentum
    
    # Volume Weighting
    volume_weighted_momentum = df['volume'] * momentum_differential
    
    # Smooth with Exponential Moving Average
    ema_momentum = volume_weighted_momentum.ewm(span=10, adjust=False).mean()
    
    # Analyze Day-to-Day Momentum Continuation
    close_open_diff = df['open'] - df['close'].shift(1)
    
    # Adjust Factor Value Based on Volume Spikes
    moving_avg_volume = df['volume'].rolling(window=20).mean()
    significant_vol_deviation = (df['volume'] - moving_avg_volume) / moving_avg_volume
    adjusted_ema_momentum = ema_momentum + significant_vol_deviation * ema_momentum.sign()
    
    # Calculate Intraday Momentum Components
    high_low_diff = df['high'] - df['low']
    open_close_momentum = df['close'] - df['open']
    
    # Combine Intraday Momentum Components
    combined_intraday_momentum = (high_low_diff + open_close_momentum) / 2
    close_std_dev = df['close'].rolling(window=5).std()
    adjusted_intraday_momentum = combined_intraday_momentum / close_std_dev
    
    # Volume Weighted Intraday Momentum
    volume_weighted_intraday_momentum = adjusted_intraday_momentum * df['volume']
    
    # Volume Confirmation
    significant_volume_increase = (df['volume'] > moving_avg_volume * 1.5)
    final_intraday_momentum = volume_weighted_intraday_momentum.where(significant_volume_increase, adjusted_intraday_momentum)
    
    # Adjust for Close-to-Open Reversal and Intraday Volatility
    close_open_reversal = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    reversal_adjusted_momentum = high_low_diff * close_open_reversal
    
    # Integrate Intraday Volatility and Reversal Adjusted Momentum
    integrated_intraday = intraday_volatility + reversal_adjusted_momentum
    
    # Final Combined Factor
    final_factor = intraday_metrics * adjusted_ema_momentum + final_intraday_momentum + integrated_intraday
    
    # Smoothing
    alpha_factor = final_factor.ewm(span=15, adjust=False).mean()
    
    return alpha_factor
