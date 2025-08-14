import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Short-Term Momentum
    short_term_momentum = df['close'] - df['close'].rolling(window=7).mean()
    
    # Calculate Long-Term Momentum
    long_term_momentum = df['close'] - df['close'].rolling(window=25).mean()
    
    # Create a Momentum Differential
    momentum_differential = long_term_momentum - short_term_momentum
    
    # Intraday Price Momentum
    high_low_diff = df['high'] - df['low']
    open_close_momentum = df['close'] - df['open']
    intraday_price_momentum = 0.6 * high_low_diff + 0.4 * open_close_momentum
    
    # Volume Weighting and Confirmation
    volume_weighted_mom_diff = df['volume'] * momentum_differential
    significant_volume_increase = df['volume'] > df['volume'].rolling(window=10).mean()
    volume_boosted_mom = volume_weighted_mom_diff * (1 + significant_volume_increase)
    
    # Adjust for Close-to-Open Reversal
    close_to_open_reversal = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    adjusted_high_low_momentum = high_low_diff * (1 - close_to_open_reversal)
    
    # Combine Intraday Momentum Components
    combined_intraday_momentum = (intraday_price_momentum + adjusted_high_low_momentum) / 2
    std_dev_close = df['close'].rolling(window=25).std()
    adjusted_combined_momentum = combined_intraday_momentum / std_dev_close
    
    # Volume and Volatility Weighting
    daily_return = df['close'] - df['close'].shift(1)
    sma_5_day_return = daily_return.rolling(window=5).mean()
    daily_volatility = df['high'] - df['low']
    sma_5_day_volatility = daily_volatility.rolling(window=5).mean()
    volume_adjusted_momentum = sma_5_day_return * df['volume']
    normalized_volume_adjusted_momentum = volume_adjusted_momentum / sma_5_day_volatility
    
    # Final Integrated Momentum
    final_integrated_momentum = volume_boosted_mom + adjusted_combined_momentum
    final_integrated_momentum = final_integrated_momentum * normalized_volume_adjusted_momentum
    
    # Analyze Day-to-Day Momentum Continuation
    yesterday_close_to_today_open = df['open'] - df['close'].shift(1)
    last_3_days_close = df['close'].shift(1) + df['close'].shift(2) + df['close'].shift(3)
    short_term_reversal = df['close'] - df['close'].shift(1) - df['close'].rolling(window=5).mean().diff()
    final_integrated_momentum += yesterday_close_to_today_open + last_3_days_close + short_term_reversal
    
    # Adjust Factor Value Based on Volume Spikes
    moving_average_volume = df['volume'].rolling(window=10).mean()
    significant_volume_spike = df['volume'] > 1.5 * moving_average_volume
    final_integrated_momentum *= (1 + (significant_volume_spike * (final_integrated_momentum > 0) - significant_volume_spike * (final_integrated_momentum < 0)))
    
    # Integrate and Smooth the Signal
    smoothed_signal = final_integrated_momentum.rolling(window=3).mean()
    
    return smoothed_signal
