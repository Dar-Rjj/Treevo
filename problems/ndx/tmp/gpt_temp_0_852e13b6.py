import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate Close-Open Spread
    close_open_spread = df['close'] - df['open']
    
    # Calculate 20-day Price Momentum
    price_momentum = df['close'] - df['close'].shift(20)
    
    # Identify Breakout Days
    avg_high_low_range_20 = high_low_range.rolling(window=20).mean()
    breakout_days = (df['high'] - df['low']) > 2 * avg_high_low_range_20
    
    # Volume-Adjusted Breakout Impact
    daily_return = (df['close'] / df['open']) - 1
    volume_adjusted_return = daily_return * df['volume']
    volume_adjusted_breakout_impact = (volume_adjusted_return * breakout_days).rolling(window=20).sum()
    
    # Intraday Momentum Indicator
    intraday_momentum = (df['close'] / df['open']) - 1
    
    # Volume-Weighted Average Price
    vwap = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3) / df['volume']
    
    # Combine High-Low Range, Volume-Adjusted Breakout Impact, and Close Price Trend
    sum_high_low_range = high_low_range.rolling(window=20).sum()
    combined_factor = sum_high_low_range + volume_adjusted_breakout_impact
    trend = np.sign(price_momentum)
    combined_factor *= np.where(trend > 0, 1.5, 1.0)
    
    # Integrate Volume Trend Impact
    volume_change = df['volume'] - df['volume'].shift(1)
    volume_trend = volume_change.ewm(span=5, adjust=False).mean()
    
    # Compute Volume and Amount Volatility
    volume_volatility = df['volume'].diff().rolling(window=20).std()
    amount_volatility = df['amount'].diff().rolling(window=20).std()
    combined_volatility = volume_volatility + amount_volatility
    combined_volatility = np.where(combined_volatility == 0, 1, combined_volatility)
    
    # Adjust Momentum by Combined Volatility
    adjusted_momentum = price_momentum / np.sqrt(combined_volatility)
    
    # Combine All Factors
    combined_factors = adjusted_momentum + volume_adjusted_breakout_impact
    combined_factors *= volume_trend
    combined_factors *= combined_factor
    
    # Calculate Volume Growth Rate
    volume_growth_rate = df['volume'] / df['volume'].shift(1)
    
    # Integrate Adjusted Momentum, Volume-Adjusted Breakout Impact, and Close-Open Spread
    integrated_result = adjusted_momentum + volume_adjusted_breakout_impact + np.sign(close_open_spread) * close_open_spread
    
    # Final Alpha Factor
    final_alpha_factor = integrated_result * volume_growth_rate
    
    # Introduce Volume Spike Indicator
    average_volume_20 = df['volume'].rolling(window=20).mean()
    volume_spike = df['volume'] > 2 * average_volume_20
    final_alpha_factor = np.where(volume_spike, final_alpha_factor * 1.5, final_alpha_factor)
    
    return final_alpha_factor
