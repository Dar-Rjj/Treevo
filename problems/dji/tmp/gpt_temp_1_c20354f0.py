import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Adjust Intraday Range by Volume
    volume_ema = df['volume'].ewm(span=10).mean()
    volume_adjusted = df['volume'] / volume_ema
    adjusted_intraday_range = intraday_range * volume_adjusted
    
    # Divide by High-Low Price Volatility
    true_range = (df[['high', 'low']].diff(axis=1).iloc[:, 1].abs() + 
                  (df['close'] - df[['high', 'low']]).abs().max(axis=1))
    high_low_volatility = true_range.rolling(window=10).std()
    adjusted_intraday_range_volatility = adjusted_intraday_range / high_low_volatility
    
    # Further Adjustment by Close Price Volatility
    close_returns = df['close'].pct_change()
    close_price_volatility = close_returns.rolling(window=10).std()
    adjusted_intraday_range_final = adjusted_intraday_range_volatility / close_price_volatility
    
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Calculate Adjusted High-Low Spread
    adjusted_high_low_spread = high_low_spread / df['close']
    
    # Calculate Intraday Return
    intraday_return = (df['high'] - df['low']) / df['low']
    
    # Combine Adjusted High-Low Spread and Intraday Return
    combined_value = adjusted_high_low_spread + intraday_return
    
    # Weight by Volume
    volume_weighted_combined_value = combined_value * df['volume']
    
    # Calculate 5-Day Average of Combined Value
    five_day_average_combined_value = volume_weighted_combined_value.rolling(window=5).mean()
    
    # Subtract 5-Day Average of Combined Value from Volume-Weighted Combined Value
    adjusted_combined_value = volume_weighted_combined_value - five_day_average_combined_value
    
    # Further Adjustment by Open Price Volatility and Return Momentum
    open_log_returns = np.log(df['open']).diff()
    open_price_volatility = open_log_returns.rolling(window=10).std()
    close_log_returns = np.log(df['close']).diff()
    return_momentum = close_log_returns.rolling(window=10).mean()
    
    # Final Alpha Factor
    final_alpha_factor = adjusted_intraday_range_final / open_price_volatility - return_momentum
    
    # Synthesize Intraday, High-Low, and Price-Volume Momentum
    volume_ratio = df['volume'] / df['volume'].ewm(span=10).mean()
    adjusted_intraday_range_volume = adjusted_intraday_range_final * volume_ratio
    high_low_momentum = adjusted_high_low_spread * volume_ratio
    price_momentum = close_log_returns * volume_ratio
    momentum_components_sum = adjusted_intraday_range_volume + high_low_momentum + price_momentum
    
    # Integrate All Components into a Single Alpha Factor
    integrated_alpha_factor = (adjusted_intraday_range_final + 
                               high_low_momentum + 
                               price_momentum + 
                               final_alpha_factor)
    
    # Exponential Moving Average on Integrated Alpha Factor
    ema_integrated_alpha_factor = integrated_alpha_factor.ewm(span=10).mean()
    
    return ema_integrated_alpha_factor
