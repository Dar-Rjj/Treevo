import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, market_cap, sectors):
    # Calculate Intraday Return
    df['intraday_return'] = df['high'] - df['low']
    
    # Calculate Volume-Weighted Intraday Return
    df['volume_weighted_intraday_return'] = df['intraday_return'] * df['volume']
    
    # Exponential Moving Average of Volume-Weighted Intraday Return
    df['ewma_volume_weighted_intraday_return'] = df['volume_weighted_intraday_return'].ewm(span=30, adjust=False).mean()
    
    # Calculate Intraday Volatility
    df['squared_intraday_return'] = df['intraday_return'] ** 2
    df['intraday_volatility'] = df['squared_intraday_return'].rolling(window=30).sum().apply(np.sqrt)
    
    # Simple Moving Average of Intraday Volatility
    df['sma_intraday_volatility'] = df['intraday_volatility'].rolling(window=30).mean()
    
    # Adjust for Recent Volatility
    df['recent_sma_intraday_volatility'] = df['intraday_volatility'].rolling(window=10).mean()
    
    # Calculate Volume-Weighted Intraday Momentum
    df['volume_weighted_intraday_momentum'] = df['intraday_return'] * df['volume']
    
    # Exponential Moving Average of Volume-Weighted Intraday Momentum
    df['ewma_volume_weighted_intraday_momentum'] = df['volume_weighted_intraday_momentum'].ewm(span=30, adjust=False).mean()
    
    # Calculate Stock Relative Strength
    df['return_252d'] = df['close'].pct_change(252)
    df['stock_relative_strength'] = df.groupby('date')['return_252d'].rank(pct=True)
    
    # Calculate Sector Momentum
    df['sector_avg_return_252d'] = df.groupby(['date', sectors])['return_252d'].transform('mean')
    df['sector_momentum'] = df.groupby('date')['sector_avg_return_252d'].rank(pct=True)
    
    # Adjust Volatility with Market Cap
    df['adj_intraday_volatility'] = df['intraday_volatility'] / market_cap
    
    # Combine Factors
    df['final_factor'] = (
        df['ewma_volume_weighted_intraday_return'] +
        df['sma_intraday_volatility'] +
        df['recent_sma_intraday_volatility'] +
        df['stock_relative_strength'] +
        df['sector_momentum']
    )
    
    return df['final_factor']
