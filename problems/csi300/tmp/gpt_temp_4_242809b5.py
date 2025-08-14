import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'].shift(-1) - df['close']) / df['close']
    
    # Compute Recent Weights for Dynamic Weighted Combination
    df['recent_volatility'] = df['intraday_range'].rolling(window=5).std()
    df['dynamic_weight_intraday_range'] = df['recent_volatility'].apply(lambda x: 0.8 if x > df['recent_volatility'].mean() else 0.6)
    df['dynamic_weight_close_to_open'] = df['recent_volatility'].apply(lambda x: 0.2 if x > df['recent_volatility'].mean() else 0.4)
    
    # Final Combined Factor
    df['combined_factor'] = (df['intraday_range'] * df['dynamic_weight_intraday_range']) + (df['close_to_open_return'] * df['dynamic_weight_close_to_open'])
    
    # Incorporate Price Momentum
    df['5_day_momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['momentum_integrated_factor'] = (df['combined_factor'] * 0.7) + (df['5_day_momentum'] * 0.3)
    
    # Incorporate Volume Shocks
    df['volume_shock'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['volume_shock_integrated_factor'] = (df['momentum_integrated_factor'] * 0.8) + (df['volume_shock'] * 0.2)
    
    # Incorporate Trend Analysis
    df['10_day_ma'] = df['close'].rolling(window=10).mean()
    df['trend'] = df.apply(lambda row: 0.15 if row['close'] > row['10_day_ma'] else -0.15, axis=1)
    df['final_alpha_factor'] = (df['volume_shock_integrated_factor'] * 0.85) + (df['trend'] * 0.15)
    
    return df['final_alpha_factor']
