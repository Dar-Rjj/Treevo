import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Volatility and High-Low Range
    df['intraday_volatility'] = df['high'] - df['low']
    df['high_low_range'] = df['high'] - df['low']
    
    # Weight by Volume-to-Price Ratio
    average_price = (df['high'] + df['low']) / 2
    volume_to_price_ratio = df['volume'] / average_price
    df['weighted_intraday_volatility'] = df['intraday_volatility'] * volume_to_price_ratio
    
    # Enhance with Close-to-Open Change
    close_open_change = (df['close'] - df['open']) / df['open']
    df['enhanced_factor'] = df['weighted_intraday_volatility'] - close_open_change
    
    # Calculate Momentum
    lookback_period = 10  # Example lookback period
    df['high_low_momentum'] = df['high_low_range'].rolling(window=lookback_period).mean()
    df['close_open_momentum'] = (df['close'] - df['open']).rolling(window=lookback_period).mean()
    
    # Compute Final Factor
    df['final_factor'] = df['enhanced_factor']
    df['final_factor'] = df['final_factor'].where(
        df['close_open_momentum'] > 0,
        df['final_factor'] * df['close_open_momentum'],
        df['final_factor'] / df['close_open_momentum']
    )
    
    return df['final_factor']
