import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Price
    df['vw_price'] = df['close'] * df['volume']
    
    # Calculate Daily Return
    df['daily_return'] = df['vw_price'].pct_change()
    
    # Smooth the Daily Returns using a Simple Moving Average (last 10 days)
    df['smoothed_return'] = df['daily_return'].rolling(window=10).mean()
    
    # Sum 5-Day Smoothed Returns
    df['sum_5d_smoothed_return'] = df['smoothed_return'].rolling(window=5).sum()
    
    # Adjust Momentum by Volume Trend
    df['volume_trend'] = df['volume'] / df['volume'].shift(m) - 1
    df['adjusted_momentum'] = df['sum_5d_smoothed_return'] * df['volume_trend']
    
    # Adjust Momentum by Price Volatility
    df['price_range'] = df['high'] - df['low']
    df['adjusted_momentum'] = df['adjusted_momentum'] / df['price_range']
    
    # Confirm Momentum with Volume Surge
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['confirmed_momentum'] = df['adjusted_momentum'] * (df['volume_change'] > 0)
    
    # Enhance the Factor
    df['enhanced_factor'] = df['confirmed_momentum'] * abs(df['volume_change'])
    df['enhanced_factor'] = df['enhanced_factor'].where(df['enhanced_factor'] > 2.0, 0)
    
    # Introduce Secondary Momentum Indicator
    df['sma_20'] = df['vw_price'].rolling(window=20).mean()
    df['momentum_20'] = (df['vw_price'] / df['sma_20']) - 1
    
    # Combine 5-Day and 20-Day Momentum
    df['combined_momentum'] = 0.7 * df['enhanced_factor'] + 0.3 * df['momentum_20']
    
    # Introduce Additional Momentum Confirmation
    df['sma_10'] = df['vw_price'].rolling(window=10).mean()
    df['momentum_10'] = (df['vw_price'] / df['sma_10']) - 1
    
    # Combine 5-Day, 10-Day, and 20-Day Momentum
    df['combined_momentum'] = 0.4 * df['enhanced_factor'] + 0.3 * df['momentum_10'] + 0.3 * df['momentum_20']
    
    # Final Adjustment by Price Volatility
    df['final_factor'] = df['combined_momentum'] / df['price_range']
    
    return df['final_factor'].dropna()
