import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Volatility
    intraday_volatility = df['high'] - df['low']
    
    # Weight by Volume
    weighted_volatility = intraday_volatility * df['volume']
    
    # Calculate Intraday Momentum
    intraday_momentum = df['close'] - df['open']
    
    # Integrate Momentum and Volatility
    integrated_factor = weighted_volatility + intraday_momentum
    integrated_factor = integrated_factor.ewm(span=5, adjust=False).mean()  # Exponential Smoothing
    integrated_factor = np.log1p(integrated_factor)  # Logarithmic Transformation
    
    # Incorporate Liquidity Insights
    bid_ask_spread = df['ask'] - df['bid']
    avg_bid_ask_spread = bid_ask_spread.rolling(window=5).mean()
    adjusted_factor = integrated_factor / (1 + avg_bid_ask_spread)
    
    return adjusted_factor
