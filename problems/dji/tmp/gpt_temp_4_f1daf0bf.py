import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Exponential and Logarithmic Daily Returns
    df['exp_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))
    
    # Short-Term EMA of Daily Returns
    short_window = 5
    df['ema_exp_short'] = df['exp_return'].ewm(span=short_window, adjust=False).mean()
    df['ema_log_short'] = df['log_return'].ewm(span=short_window, adjust=False).mean()
    
    # Long-Term EMA of Daily Returns
    long_window = 20
    df['ema_exp_long'] = df['exp_return'].ewm(span=long_window, adjust=False).mean()
    df['ema_log_long'] = df['log_return'].ewm(span=long_window, adjust=False).mean()
    
    # Compute the Dynamic Difference for Both Return Types
    df['mom_osc_exp'] = df['ema_exp_short'] - df['ema_exp_long']
    df['mom_osc_log'] = df['ema_log_short'] - df['ema_log_long']
    
    # Adjust for Volume-Weighted Activity
    df['vwma'] = (df['close'] * df['volume']).rolling(window=long_window).sum() / df['volume'].rolling(window=long_window).sum()
    df['vol_weighted_mom'] = df['close'] - df['vwma']
    
    # Enhanced Volatility Component
    df['tr'] = df[['high', 'low']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1)), abs(x['low'] - df['close'].shift(1))), axis=1)
    df['atr_short'] = df['tr'].rolling(window=short_window).mean()
    df['atr_long'] = df['tr'].rolling(window=long_window).mean()
    df['vol_diff'] = df['atr_short'] - df['atr_long']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['mom_osc_exp'] + df['mom_osc_log'] + df['vol_weighted_mom'] + df['vol_diff']
    
    return df['alpha_factor']
