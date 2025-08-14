import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Short-Term Logarithmic Return
    short_term_return = np.log(df['close'] / df['close'].shift(5))
    
    # Calculate Long-Term Logarithmic Return
    long_term_return = np.log(df['close'] / df['close'].shift(20))
    
    # Calculate Volume-Weighted Short-Term Logarithmic Return
    volume_weighted_short_term_return = short_term_return * df['volume']
    
    # Calculate Volume-Weighted Long-Term Logarithmic Return
    volume_weighted_long_term_return = long_term_return * df['volume']
    
    # Calculate Short-Term Volatility (Average True Range)
    high_low_range = df['high'] - df['low']
    short_term_volatility = high_low_range.rolling(window=5).mean()
    
    # Adjust for Volatility
    adjusted_return = (volume_weighted_short_term_return / short_term_volatility) - volume_weighted_long_term_return
    
    # Determine Gain and Loss
    gain = np.where(df['close'] > df['close'].shift(1), df['close'] - df['close'].shift(1), 0)
    loss = np.where(df['close'] < df['close'].shift(1), df['close'].shift(1) - df['close'], 0)
    
    # Aggregate Gains and Losses over a period (e.g., 14 days)
    sum_gains = gain.rolling(window=14).sum()
    sum_losses = loss.rolling(window=14).sum()
    
    # Calculate Relative Strength
    rs = sum_gains / sum_losses
    
    # Convert to ARSI
    rsi = 100 - (100 / (1 + rs))
    arsi = rsi * df['volume'].rolling(window=14).mean() * (df['close'] / df['open'])
    
    # Calculate Intraday Volatility
    intraday_volatility = df['high'] - df['low']
    
    # Calculate Close-to-Close Return
    close_to_close_return = (df['close'] / df['close'].shift(1)) - 1
    
    # Combine Intraday Volatility and Close-to-Close Return
    combined_volatility_return = intraday_volatility * close_to_close_return
    
    # Calculate Daily Volume Change
    daily_volume_change = (df['volume'] / df['volume'].shift(1)) - 1
    
    # Aggregate Volume Changes over the last M days (M = 5)
    aggregated_volume_changes = daily_volume_change.rolling(window=5).sum()
    
    # Calculate Price Oscillator
    price_oscillator = short_term_return - long_term_return
    
    # Calculate Price Range Ratio
    price_range_ratio = df['high'] / df['low']
    
    # Combine ARSI, Adjusted Returns, and Price Oscillator
    combined_factor = (arsi * adjusted_return) + price_oscillator - volume_weighted_long_term_return
    
    # Final Adjustment
    final_adjustment = (combined_factor + adjusted_return + combined_volatility_return) * aggregated_volume_changes * price_range_ratio
    
    return final_adjustment
