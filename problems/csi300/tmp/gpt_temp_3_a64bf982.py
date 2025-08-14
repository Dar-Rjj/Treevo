import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday High-to-Low Range
    df['daily_range'] = df['high'] - df['low']
    df['intraday_high_low_range'] = df['daily_range'] / df['open']
    
    # Open to Close Momentum
    df['open_to_close_return'] = (df['close'] / df['open']) - 1
    df['smoothed_open_to_close_momentum'] = df['open_to_close_return'].rolling(window=5).mean()
    
    # Volume Adjusted Intraday Movement
    avg_volume_20_days = df['volume'].rolling(window=20).mean()
    df['volume_adjusted_intraday_movement'] = (df['close'] - df['open']) / avg_volume_20_days
    
    # Price-Volume Trend Indicator
    df['daily_price_change'] = df['close'] - df['close'].shift(1)
    df['price_volume_trend'] = (df['daily_price_change'] * df['volume']).rolling(window=30).sum()
    
    # Sector-Specific Volatility
    daily_returns = df['close'].pct_change()
    df['daily_volatility'] = daily_returns.rolling(window=20).std()
    df['sector_specific_volatility'] = df['daily_volatility'].rolling(window=20).mean()
    
    # Leverage Weighted Returns
    # Assuming debt/equity ratio is a column in the dataframe named 'leverage'
    df['leverage_weighted_returns'] = (df['close'] / df['close'].shift(1) - 1) * df['leverage']
    df['leverage_weighted_returns_agg'] = df['leverage_weighted_returns'].rolling(window=30).sum()
    
    # Liquidity Measure
    # Assuming shares outstanding is a column in the dataframe named 'shares_outstanding'
    df['turnover_ratio'] = df['volume'] / df['shares_outstanding']
    df['liquidity_measure'] = df['turnover_ratio'].rolling(window=20).mean()
    
    # Combined Momentum and Volatility Factor
    combined_momentum_volatility = (df['intraday_high_low_range'] * 0.3 + 
                                    df['smoothed_open_to_close_momentum'] * 0.3 + 
                                    df['sector_specific_volatility'] * 0.4)
    df['combined_momentum_volatility'] = combined_momentum_volatility.ewm(alpha=0.2).mean()
    
    # Volume-Sensitive Momentum Factor
    volume_sensitive_momentum = (df['price_volume_trend'] * 0.6 + 
                                 df['volume_adjusted_intraday_movement'] * 0.4)
    df['volume_sensitive_momentum'] = volume_sensitive_momentum.ewm(alpha=0.2).mean()
    
    # Enhanced Liquidity-Momentum Factor
    enhanced_liquidity_momentum = (df['leverage_weighted_returns_agg'] * 0.7 + 
                                   df['liquidity_measure'] * 0.3)
    df['enhanced_liquidity_momentum'] = enhanced_liquidity_momentum.ewm(alpha=0.2).mean()
    
    # Final Alpha Factor
    final_alpha_factor = (df['combined_momentum_volatility'] * 0.4 + 
                          df['volume_sensitive_momentum'] * 0.3 + 
                          df['enhanced_liquidity_momentum'] * 0.3)
    df['final_alpha_factor'] = final_alpha_factor.ewm(alpha=0.2).mean()
    
    return df['final_alpha_factor']
