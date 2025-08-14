import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Enhanced Intraday High-Low Spread
    df['high_low_spread'] = (df['high'] - df['low']) * df['volume']
    
    # Combine Intraday High-Low Spread and Intraday Return
    df['combined_intraday_factor'] = df['high_low_spread'] * df['intraday_return']
    
    # Incorporate Volume and Amount Influence
    avg_volume = df['volume'].rolling(window=20).mean()
    df['volume_impact'] = df['volume'] / avg_volume
    avg_amount = df['amount'].rolling(window=20).mean()
    df['amount_impact'] = df['amount'] / avg_amount
    df['volume_amount_impact'] = df['volume_impact'] + df['amount_impact']
    df['weighted_intraday_factor'] = df['combined_intraday_factor'] * df['volume_amount_impact']
    
    # Calculate True Range for each day
    df['true_range'] = df.apply(
        lambda row: max(row['high'] - row['low'], 
                        abs(row['high'] - df.shift(1)['close'][row.name]), 
                        abs(row['low'] - df.shift(1)['close'][row.name])), axis=1)
    
    # Calculate 14-day Simple Moving Average of the True Range
    df['tr_sma_14'] = df['true_range'].rolling(window=14).mean()
    
    # Construct the Momentum Factor
    df['momentum_component'] = (df['close'] - df['tr_sma_14']) / df['tr_sma_14']
    
    # Calculate Daily Log Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Compute Daily Volatility
    df['volatility'] = df['log_return'].rolling(window=20).std()
    
    # Normalize Momentum by Volatility
    df['normalized_momentum'] = df['momentum_component'] / df['volatility']
    
    # Introduce Trend Component
    df['trend_ma_50'] = df['close'].rolling(window=50).mean()
    df['trend_direction'] = np.where(df['close'] > df['trend_ma_50'], 1, -1)
    
    # Final Alpha Factor
    df['alpha_factor'] = df['normalized_momentum'] + df['weighted_intraday_factor']
    df['alpha_factor'] = df['alpha_factor'] * df['trend_direction']
    
    return df['alpha_factor']
