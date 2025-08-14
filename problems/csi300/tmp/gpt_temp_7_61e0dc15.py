import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Price Range
    intraday_price_range = df['high'] - df['low']
    
    # Compute Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Determine Intraday Volatility
    intraday_volatility = df[['high', 'low', 'open']].apply(lambda x: max((x['high'] - x['low']), abs(x['close'] - x['open'])), axis=1)
    
    # Adjust Volatility by Volume
    adjusted_volatility = intraday_volatility * df['volume']
    
    # Calculate Intraday Momentum
    intraday_momentum = df['close'] - df['open'].shift(1)
    
    # Calculate Intraday Trading Imbalance
    intraday_trading_imbalance = df['amount'] / df['volume']
    
    # Combine Intraday Return, Adjusted Volatility, Momentum, and Trading Imbalance
    combined_value = (intraday_return 
                      - adjusted_volatility 
                      + intraday_momentum) * intraday_trading_imbalance
    
    # Generate Final Alpha Factor
    alpha_factor = (combined_value > 0).astype(int)
    
    return alpha_factor
