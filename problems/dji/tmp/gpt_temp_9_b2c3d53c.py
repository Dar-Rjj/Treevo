import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = df['close'] / df['open'] - 1
    
    # Calculate Reversal Signal
    df['reversal_signal'] = np.where(df['intraday_return'] > 0, -1, 1)
    
    # Calculate Daily Price Return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate 20-Day Weighted Moving Average of Returns
    weighted_returns = (df['daily_return'] * df['volume']).rolling(window=20).sum()
    total_volume = df['volume'].rolling(window=20).sum()
    df['weighted_moving_avg_return'] = weighted_returns / total_volume
    
    # Adjust for Price Volatility
    df['price_range'] = df['high'] - df['low']
    average_price_range = df['price_range'].rolling(window=20).mean()
    df['final_factor_value'] = df['weighted_moving_avg_return'] - average_price_range
    
    # Combine Intraday and Volume-Adjusted Momentum
    # Compute Average True Range (ATR)
    df['tr'] = df[['high' - 'low', abs('high' - df['close'].shift(1)), abs('low' - df['close'].shift(1))]].max(axis=1)
    df['atr'] = df['tr'].rolling(window=20).mean()
    
    # Multiply ATR by Reversal Signal
    df['atr_reversal'] = df['atr'] * df['reversal_signal']
    
    # Add to 20-Day Weighted Moving Average of Returns
    df['combined_factor'] = df['atr_reversal'] + df['weighted_moving_avg_return']
    
    # Smooth Result with Exponential Moving Average (EMA) with a short period
    df['alpha_factor'] = df['combined_factor'].ewm(span=5, adjust=False).mean()
    
    return df['alpha_factor']
