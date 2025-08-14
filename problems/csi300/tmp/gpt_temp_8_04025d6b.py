import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute Intraday Momentum Intensity
    high_low_ratio = df['high'] / df['low']
    open_close_diff = df['close'] - df['open']
    intraday_momentum_intensity = (0.6 * (df['high'] - df['low'])) + (0.4 * (df['close'] - df['open']))
    
    # Analyze Day-to-Day Momentum Continuation
    day_to_day_momentum = df['open'].shift(-1) - df['close']
    last_3_days_avg = df['close'].rolling(window=3).mean()
    
    # Calculate Short-Term and Long-Term Momentum
    short_term_momentum = df['close'] - df['close'].rolling(window=7).mean()
    long_term_momentum = df['close'] - df['close'].rolling(window=25).mean()
    
    # Create a Momentum Differential
    momentum_differential = long_term_momentum - short_term_momentum
    
    # Volume Weighting and Confirmation
    volume_weighted_momentum = momentum_differential * df['volume']
    avg_volume_20 = df['volume'].rolling(window=20).mean()
    significant_volume_increase = df['volume'] > avg_volume_20
    weighted_momentum = momentum_differential.where(~significant_volume_increase, momentum_differential * 1.5)
    
    # Integrate Signals
    integrated_signal = weighted_momentum + intraday_momentum_intensity
    
    # Adjust Factor Value Based on Volume Spikes
    moving_avg_volume = df['volume'].rolling(window=20).mean()
    significant_volume_deviation = df['volume'] > moving_avg_volume * 1.5
    factor_adjustment = 1.5 if significant_volume_deviation & (integrated_signal > 0) else 0.5 if significant_volume_deviation & (integrated_signal < 0) else 1
    final_factor = integrated_signal * factor_adjustment
    
    return final_factor
