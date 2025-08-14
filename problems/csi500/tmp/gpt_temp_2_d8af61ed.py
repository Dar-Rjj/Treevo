import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Returns
    df['daily_returns'] = df['close'].pct_change()
    
    # Calculate 10-day Moving Average of Returns
    df['ma_10_returns'] = df['daily_returns'].rolling(window=10).mean()
    
    # Calculate Price Volatility
    df['price_volatility'] = (df['high'] - df['low']) / df['close']
    
    # Calculate Volume Spike Factor
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = df['volume'] > 1.5 * df['volume_ma_20']
    df['adjusted_ma_10'] = df['ma_10_returns'] * (0.5 if df['volume_spike'] else 1)
    
    # Calculate High Price Volatility Factor
    df['high_price_volatility'] = df['price_volatility'] > 0.05
    df['adjusted_ma_10'] = df['adjusted_ma_10'] * (0.8 if df['high_price_volatility'] else 1)
    
    # Calculate Amount Change
    df['amount_change'] = df['amount'].pct_change()
    
    # Adjust for Sudden Amount Increase
    df['sudden_amount_increase'] = df['amount'] > 1.5 * df['amount'].shift(1)
    df['final_adjusted_momentum'] = df['adjusted_ma_10'] * (0.7 if df['sudden_amount_increase'] else 1)
    
    return df['final_adjusted_momentum']
