import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Range
    df['price_range'] = df['high'] - df['low']
    
    # Compute Volume-Adjusted Momentum
    df['momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['volume_adjusted_momentum'] = df['momentum'] * df['volume']
    
    # Integrate High-Low Spread and Volume-Adjusted Momentum
    df['avg_price_range_10'] = df['price_range'].rolling(window=10).mean()
    df['vol_adj_mom_with_spread'] = (df['volume_adjusted_momentum'] * df['price_range']) / df['avg_price_range_10']
    
    # Apply Directional Bias
    df['directional_bias'] = 1.5 if df['close'] > df['open'] else 0.5
    df['vol_adj_mom_with_spread'] = df['vol_adj_mom_with_spread'] * df['directional_bias']
    
    # Incorporate Open-Close Trend
    df['open_close_trend'] = df['close'] - df['open']
    df['trend_adjusted_factor'] = df['vol_adj_mom_with_spread'] * (1.2 if df['open_close_trend'] > 0 else 0.8)
    
    # Compute Enhanced Intraday Volatility
    df['enhanced_intraday_volatility'] = df['price_range'] * 1.5
    
    # Weight by Volume Adjusted Factor
    df['avg_volume_20'] = df['volume'].rolling(window=20).mean()
    df['volume_weight'] = df['volume'] / df['avg_volume_20']
    
    # Integrate with Closing Price Momentum
    df['closing_price_momentum'] = df['close'] - df['close'].shift(1)
    df['weighted_closing_price_momentum'] = df['closing_price_momentum'] * df['volume_weight']
    
    # Construct Final Alpha Factor
    df['intraday_factors_sum'] = (df['price_range'] * df['volume_weight'] + 
                                   df['enhanced_intraday_volatility'] + 
                                   df['weighted_closing_price_momentum'])
    df['final_alpha_factor'] = df['intraday_factors_sum'] * df['trend_adjusted_factor']
    df['final_alpha_factor'] = df['final_alpha_factor'].rolling(window=7).mean()
    
    return df['final_alpha_factor']
