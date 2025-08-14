import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday High-to-Low Range
    daily_range = (df['high'] - df['low']) / df['open']
    
    # Open to Close Momentum
    open_to_close_return = (df['close'] - df['open']) / df['open']
    open_to_close_sma = open_to_close_return.rolling(window=5).mean()
    
    # Volume Adjusted Intraday Movement
    intraday_movement = (df['close'] - df['open'])
    avg_volume_20 = df['volume'].rolling(window=20).mean()
    volume_adjusted_movement = intraday_movement / avg_volume_20
    
    # Price-Volume Trend Indicator
    price_change = df['close'] - df['close'].shift(1)
    price_volume_trend = (price_change * df['volume']).rolling(window=30).sum()
    
    # Combined Momentum and Volatility Factor
    recent_volatility = df['close'].pct_change().rolling(window=20).std()
    recent_momentum = open_to_close_return.rolling(window=20).mean()
    weight_intraday_range = 1 / (1 + recent_volatility)
    weight_open_to_close = 1 / (1 + 1/recent_momentum)
    combined_factor = (weight_intraday_range * daily_range) + (weight_open_to_close * open_to_close_sma)
    combined_factor_smoothed = combined_factor.ewm(alpha=0.2).mean()
    
    # Volume-Sensitive Momentum Factor
    recent_volume = df['volume'].rolling(window=20).mean()
    weight_price_volume = 1 / (1 + 1/recent_volume)
    weight_volume_adjusted = 1 / (1 + recent_volume)
    volume_sensitive_factor = (weight_price_volume * price_volume_trend) + (weight_volume_adjusted * volume_adjusted_movement)
    volume_sensitive_factor_smoothed = volume_sensitive_factor.ewm(alpha=0.2).mean()
    
    # Bid-Ask Spread Feature
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    bid_ask_spread = (df['ask'] - df['bid']) / typical_price
    
    # Final Alpha Factor
    recent_market_conditions = (recent_volatility + recent_momentum) / 2
    weight_combined_factor = 1 / (1 + 1/recent_market_conditions)
    weight_volume_sensitive = 1 / (1 + recent_market_conditions)
    final_alpha = (weight_combined_factor * combined_factor_smoothed) + (weight_volume_sensitive * volume_sensitive_factor_smoothed) + bid_ask_spread
    final_alpha_smoothed = final_alpha.ewm(alpha=0.2).mean()
    
    return final_alpha_smoothed
