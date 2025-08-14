import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Price
    df['volume_weighted_price'] = df['close'] * df['volume']
    
    # Calculate True Range
    df['tr'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
    df['true_range'] = (df['tr'] + df['high'].shift(1) - df['low'].shift(1)).max(axis=1)
    
    # Calculate Momentum using Volume-Weighted Price
    momentum_window = 20
    df['momentum'] = df['volume_weighted_price'] / df['volume_weighted_price'].shift(momentum_window) - 1
    
    # Adjust Momentum by True Range
    df['adjusted_momentum'] = df['momentum'] / df['true_range']
    
    # Incorporate Trend Component
    trend_window = 50
    df['wma'] = df['close'].ewm(span=trend_window, adjust=False).mean()
    df['trend_adjusted_momentum'] = df['adjusted_momentum'] * (df['close'] / df['wma'])
    
    # Calculate Daily Log Return
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Sum Daily Log Returns over 10 days
    log_return_window = 10
    df['sum_log_return'] = df['log_return'].rolling(window=log_return_window, min_periods=1).sum()
    
    # Confirm Momentum with Volume and Amount
    volume_change_ratio_threshold = 1.05
    amount_change_ratio_threshold = 1.05
    
    df['volume_change_ratio'] = df['volume'] / df['volume'].shift(1)
    df['amount_change_ratio'] = df['amount'] / df['amount'].shift(1)
    
    # If Volume or Amount change ratio is below threshold, set factor to zero
    df.loc[df['volume_change_ratio'] < volume_change_ratio_threshold, 'trend_adjusted_momentum'] = 0
    df.loc[df['amount_change_ratio'] < amount_change_ratio_threshold, 'trend_adjusted_momentum'] = 0
    
    # Adjust Momentum by Enhanced Volatility and Volume Trend
    atr_window = 14
    df['atr'] = df['true_range'].rolling(window=atr_window, min_periods=1).mean()
    df['atr_ratio'] = df['atr'] / df['atr'].shift(atr_window)
    df['volatility_adjusted_momentum'] = df['trend_adjusted_momentum'] / df['atr_ratio']
    
    # Calculate Volume Trend
    volume_trend_window = 30
    df['volume_ema'] = df['volume'].ewm(span=volume_trend_window, adjust=False).mean()
    df['volume_trend_ratio'] = df['volume'] / df['volume_ema']
    
    # Further adjust by Volume Trend Ratio
    df['final_factor'] = df['volatility_adjusted_momentum'] * df['volume_trend_ratio']
    
    return df['final_factor']

# Example usage:
# data = pd.read_csv('your_data.csv', parse_dates=True, index_col='date')
# factor_values = heuristics_v2(data)
