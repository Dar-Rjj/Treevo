import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate Price Breakout Ratio
    price_breakout_ratio = (df['high'] - df['open']) / high_low_range
    
    # Calculate Volume Breakout Indicator
    volume_breakout_indicator = df['volume'] * (df['close'] - df['open'])
    
    # Aggregate Breakout Indicators
    volume_weighted_breakout = (price_breakout_ratio * df['volume']).cumsum()
    
    # Generate Daily Price Momentum
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    
    # Subtract the 26-day EMA from the 12-day EMA
    macd = ema_12 - ema_26
    
    # Adjust by Volume-Weighted Average True Range
    true_range = df[['high', 'low']].diff(axis=1).iloc[:, 0].abs() + df[['high', 'low']].diff(axis=1).iloc[:, 1].abs() + (df['close'].shift(1) - df['low']).abs()
    true_range = true_range.max(axis=1)
    atr_14 = true_range.rolling(window=14).mean()
    avg_volume_change_14 = df['volume'].diff().rolling(window=14).sum().abs()
    adjusted_true_range = atr_14 * avg_volume_change_14
    
    # Combine Momentum and Adjusted Volatility
    combined_momentum_volatility = macd / adjusted_true_range
    
    # Smooth with Exponential Moving Average
    smoothed_combined_momentum_volatility = combined_momentum_volatility.ewm(span=14, adjust=False).mean()
    
    # Calculate 14-day Close Price Average
    close_price_14_avg = df['close'].rolling(window=14).mean()
    
    # Compute Price Momentum
    price_momentum = df['close'] - close_price_14_avg
    
    # Adjust for Volume Impact
    current_volume = df['volume']
    avg_volume_14 = df['volume'].rolling(window=14).mean()
    volume_change_ratio = current_volume / avg_volume_14
    price_momentum_adjusted_volume = price_momentum * volume_change_ratio
    
    # Adjust for Amount Impact
    current_amount = df['amount']
    avg_amount_14 = df['amount'].rolling(window=14).mean()
    amount_change_ratio = current_amount / avg_amount_14
    price_momentum_adjusted = price_momentum_adjusted_volume * amount_change_ratio
    
    # Generate Final Alpha Factor
    final_alpha_factor = smoothed_combined_momentum_volatility + volume_weighted_breakout + price_momentum_adjusted
    
    return final_alpha_factor
