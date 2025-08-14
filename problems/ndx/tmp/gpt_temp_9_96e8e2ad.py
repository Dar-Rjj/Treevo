import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Short-Term Logarithmic Return
    short_term_log_return = np.log(df['close'] / df['close'].shift(5))
    
    # Calculate Long-Term Logarithmic Return
    long_term_log_return = np.log(df['close'] / df['close'].shift(20))
    
    # Calculate Volume-Weighted Short-Term Logarithmic Return
    volume_weighted_short_term_log_return = df['volume'] * short_term_log_return
    
    # Calculate Volume-Weighted Long-Term Logarithmic Return
    volume_weighted_long_term_log_return = df['volume'] * long_term_log_return
    
    # Calculate Short-Term Volatility
    high_low_range = df['high'] - df['low']
    short_term_volatility = high_low_range.rolling(window=5).mean()
    
    # Adjust for Volatility
    adjusted_for_volatility = (volume_weighted_short_term_log_return / short_term_volatility) - volume_weighted_long_term_log_return
    
    # Calculate Intraday Volatility
    intraday_volatility = df['high'] - df['low']
    
    # Calculate Close-to-Close Return
    close_to_close_return = df['close'] / df['close'].shift(1) - 1
    
    # Combine Intraday Volatility and Close-to-Close Return
    combined_intraday_volatility = intraday_volatility * close_to_close_return
    
    # Calculate Daily Volume Change
    daily_volume_change = df['volume'] - df['volume'].shift(1)
    
    # Aggregate Volume Changes
    aggregated_volume_changes = daily_volume_change.rolling(window=5).sum()
    
    # Calculate Intraday Momentum
    intraday_momentum = df['close'] / df['open'] - 1
    
    # Adjust for Intraday Momentum
    adjusted_for_intraday_momentum = (adjusted_for_volatility + combined_intraday_volatility) * aggregated_volume_changes + intraday_momentum
    
    # Final Adjustment
    final_adjustment = adjusted_for_intraday_momentum * aggregated_volume_changes
    
    # Calculate Price Difference
    price_difference = df['high'] - df['low']
    
    # Determine Gain and Loss
    gain = np.where(df['close'] > df['close'].shift(1), df['close'] - df['close'].shift(1), 0)
    loss = np.where(df['close'] < df['close'].shift(1), df['close'].shift(1) - df['close'], 0)
    
    # Aggregate Gains and Losses
    sum_of_gains = gain.rolling(window=14).sum()
    sum_of_losses = loss.rolling(window=14).sum()
    
    # Calculate Relative Strength
    relative_strength = sum_of_gains / sum_of_losses
    
    # Convert to ARSI
    rsi = 100 - (100 / (1 + relative_strength))
    arsi = rsi * df['volume'].rolling(window=14).mean() * (df['close'] / df['open'] - 1)
    
    # Synthesize ARSI with Volume-Weighted Returns and Volatility
    synthesized_arsi = arsi * volume_weighted_short_term_log_return / short_term_volatility - volume_weighted_long_term_log_return
    
    # Introduce a new factor: Volume-Weighted Intraday Volatility
    volume_weighted_intraday_volatility = df['volume'] * intraday_volatility
    
    # Combine all factors into the final alpha factor
    alpha_factor = final_adjustment + synthesized_arsi + volume_weighted_intraday_volatility
    
    return alpha_factor
