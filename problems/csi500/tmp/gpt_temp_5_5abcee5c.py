import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['close']
    
    # Compute Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']).abs().rolling(window=20).sum()
    
    # Intraday Momentum Factor
    df['intraday_momentum'] = (df['intraday_return'] * df['volume']).rolling(window=5).sum()
    
    # Volume-Weighted Intraday Reversal
    df['intraday_reversal'] = -(df['intraday_return'] * df['volume']).rolling(window=10).sum()
    
    # Calculate Price Momentum
    df['price_momentum_close'] = df['close'].pct_change(periods=20)
    df['price_momentum_open'] = df['open'].pct_change(periods=20)
    df['price_momentum'] = df['price_momentum_close'] + df['price_momentum_open']
    
    # Calculate Volume Adjusted Volatility
    df['volatility'] = (df['high'] - df['low']).std(ddof=0).rolling(window=20).mean()
    df['volume_adjusted_volatility'] = df['volatility'] / df['volume'].rolling(window=20).mean()
    
    # Combine Factors
    df['alpha_factor'] = (df['intraday_momentum'] + df['intraday_reversal']) * df['price_momentum'] * df['volume_adjusted_volatility']
    
    return df['alpha_factor']
