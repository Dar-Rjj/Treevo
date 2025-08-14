import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread Ratio
    high_low_spread_ratio = (df['high'] - df['low']) / df['close']
    
    # Calculate Price Change
    price_change = df['close'] - df['open']
    
    # Volume-Weighted Price Change
    volume_weighted_price_change = price_change * df['volume']
    
    # Calculate Average True Range (ATR)
    tr = np.maximum.reduce([df['high'] - df['low'], 
                            abs(df['high'] - df['close'].shift()), 
                            abs(df['low'] - df['close'].shift())])
    atr = tr.rolling(window=14).mean()
    
    # Intraday Volatility
    intraday_volatility = (df['high'] - df['low']) / atr
    
    # Relative Strength Line
    relative_strength_line = df['close'].rolling(window=10).mean()
    
    # Combine Factors
    alpha_factor = (high_low_spread_ratio + 
                    volume_weighted_price_change + 
                    intraday_volatility + 
                    relative_strength_line)
    
    return alpha_factor
