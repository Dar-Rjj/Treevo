import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Directionally Adjusted High-Low Spread
    df['adj_high_low_spread'] = (df['High'] - df['Low']) * 1.5 * (df['Close'] > df['Open']).astype(int) + \
                                (df['High'] - df['Low']) * 0.5 * (df['Close'] <= df['Open']).astype(int)
    
    # Calculate Volume-Weighted Return
    df['volume_weighted_return'] = (df['High'] - df['Low']) / df['Low'] * df['Volume']
    volume_ema = df['Volume'].rolling(window=14).mean()
    df['volume_weighted_return'] = df['volume_weighted_return'] * (df['Volume'] > 1.5 * volume_ema).astype(int) * 2 + \
                                  df['volume_weighted_return'] * (df['Volume'] <= 1.5 * volume_ema).astype(int)
    
    # Integrate Adjusted High-Low Spread and Volume-Weighted Return
    df['integrated_factor'] = df['adj_high_low_spread'] * df['volume_weighted_return']
    
    # Calculate Volume-Adjusted Momentum
    df['volume_adjusted_momentum'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * df['Volume']
    
    # Calculate Volume-Weighted Intraday High-Low Spread
    df['volume_weighted_intraday_spread'] = (df['High'] - df['Low']) * df['Volume']
    
    # Calculate Volume-Adjusted Opening Gap
    df['volume_adjusted_opening_gap'] = (df['Open'] - df['Close'].shift(1)) * df['Volume']
    
    # Combine Weighted Intraday High-Low Spread with Volume-Adjusted Opening Gap
    df['combined_value'] = df['volume_weighted_intraday_spread'] + df['volume_adjusted_opening_gap']
    
    # Calculate Short-Term and Long-Term EMAs of Combined Value
    df['ema_12'] = df['combined_value'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['combined_value'].ewm(span=26, adjust=False).mean()
    
    # Calculate Divergence
    df['divergence'] = df['ema_12'] - df['ema_26']
    
    # Apply Sign Function
    df['sign_divergence'] = df['divergence'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # Calculate Momentum over Moving Window
    df['momentum'] = df['integrated_factor'].rolling(window=10).sum()
    
    # Incorporate Price Range for Volatility Adjustment
    df['daily_price_range'] = df['High'] - df['Low']
    avg_price_range = df['daily_price_range'].rolling(window=10).mean()
    df['adjusted_momentum'] = df['volume_adjusted_momentum'] / avg_price_range
    
    # Combine Metrics
    df['final_factor'] = (df['adjusted_momentum'] + df['sign_divergence'] * 0.7 + df['momentum'] * 0.3)
    
    return df['final_factor'].dropna()
