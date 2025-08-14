import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = df['high'] - df['low']
    
    # Calculate Volume-Weighted Intraday Return
    volume_weighted_intraday_return = intraday_return * df['volume']
    
    # Exponential Moving Average of Volume-Weighted Intraday Return
    ema_volume_weighted_intraday_return = volume_weighted_intraday_return.ewm(span=30, adjust=False).mean()
    
    # Calculate Intraday Volatility
    squared_intraday_return = intraday_return ** 2
    intraday_volatility = (squared_intraday_return.rolling(window=30).sum()) ** 0.5
    
    # Simple Moving Average of Intraday Volatility
    sma_intraday_volatility = intraday_volatility.rolling(window=30).mean()
    
    # Adjust for Recent Volatility
    recent_volatility = intraday_volatility.rolling(window=10).mean()
    
    # Calculate Volume-Weighted Intraday Momentum
    volume_weighted_intraday_momentum = intraday_return * df['volume']
    
    # Exponential Moving Average of Volume-Weighted Intraday Momentum
    ema_volume_weighted_intraday_momentum = volume_weighted_intraday_momentum.ewm(span=30, adjust=False).mean()
    
    # Incorporate Volume Trends
    volume_trend = df['volume'].ewm(alpha=0.2, adjust=False).mean()
    
    # Adjust Volume-Weighted Intraday Return and Momentum
    adjusted_ema_volume_weighted_intraday_return = ema_volume_weighted_intraday_return / volume_trend
    adjusted_ema_volume_weighted_intraday_momentum = ema_volume_weighted_intraday_momentum / volume_trend
    
    # Adjust for Market Conditions
    market_intraday_return = df['market_high'] - df['market_low']
    market_squared_intraday_return = market_intraday_return ** 2
    market_intraday_volatility = (market_squared_intraday_return.rolling(window=30).sum()) ** 0.5
    market_ema_intraday_return = market_intraday_return.ewm(span=30, adjust=False).mean()
    market_sma_intraday_volatility = market_intraday_volatility.rolling(window=30).mean()
    
    # Adjust Individual Factors
    adjusted_ema_intraday_return = adjusted_ema_volume_weighted_intraday_return - market_ema_intraday_return
    adjusted_sma_intraday_volatility = sma_intraday_volatility - market_sma_intraday_volatility
    adjusted_recent_volatility = recent_volatility - market_sma_intraday_volatility
    
    # Combine Factors
    final_factor = (adjusted_ema_intraday_return + 
                    adjusted_sma_intraday_volatility + 
                    adjusted_recent_volatility)
    
    return final_factor
