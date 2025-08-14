import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Calculate Price Momentum
    price_momentum = (df['close'].shift(1) - df['close']) / df['close'].shift(1)
    
    # Calculate Volume-Adjusted Momentum
    volume_adjusted_momentum = price_momentum * np.sqrt(df['volume'])
    
    # Combine Spread and Volume-Adjusted Momentum
    combined_factor = high_low_spread + volume_adjusted_momentum
    
    # Volume Smoothing Adjustment
    avg_volume_5d = df['volume'].rolling(window=5).mean()
    volume_smoothing_adjusted_factor = combined_factor / avg_volume_5d
    
    # Final Volume Smoothed Factor
    final_volume_smoothed_factor = volume_smoothing_adjusted_factor / df['volume']
    
    # Calculate Price Difference for Intraday Volatility
    price_diff = df['high'] - df['low']
    
    # Determine Gain and Loss
    gain = np.where(df['close'] > df['close'].shift(1), df['close'] - df['close'].shift(1), 0)
    loss = np.where(df['close'] < df['close'].shift(1), df['close'].shift(1) - df['close'], 0)
    
    # Aggregate Gains and Losses
    sum_gains = gain.rolling(window=14).sum()
    sum_losses = loss.rolling(window=14).sum()
    
    # Calculate Relative Strength
    rs = sum_gains / sum_losses
    
    # Convert to ARSI
    rsi = 100 - (100 / (1 + rs))
    arsi = rsi * avg_volume_5d * (df['close'] / df['open'])
    
    # Calculate Short-Term Return
    short_term_return = (df['close'].shift(1) - df['close'].shift(5)) / df['close'].shift(5)
    
    # Calculate Long-Term Return
    long_term_return = (df['close'].shift(1) - df['close'].shift(20)) / df['close'].shift(20)
    
    # Calculate Volume-Weighted Short-Term Return
    volume_weighted_short_term_return = df['volume'] * short_term_return
    
    # Calculate Volume-Weighted Long-Term Return
    volume_weighted_long_term_return = df['volume'] * long_term_return
    
    # Calculate Short-Term Volatility
    true_range = df[['high', 'low']].diff().abs().max(axis=1)
    short_term_volatility = true_range.rolling(window=5).mean()
    
    # Adjust for Volatility
    adjusted_volatility = (volume_weighted_short_term_return / short_term_volatility) - volume_weighted_long_term_return
    
    # Combine Intraday Volatility and Close-to-Close Return
    intraday_volatility = df['high'] - df['low']
    close_to_close_return = (df['close'] / df['close'].shift(1)) - 1
    combined_volatility_return = intraday_volatility * close_to_close_return
    
    # Final Adjustment
    final_adjustment = adjusted_volatility + combined_volatility_return
    
    # Final Alpha Factor
    alpha_factor = final_volume_smoothed_factor + final_adjustment
    
    return alpha_factor
