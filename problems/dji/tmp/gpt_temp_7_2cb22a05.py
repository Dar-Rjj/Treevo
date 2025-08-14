import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Range
    intraday_range = df['high'] - df['low']
    
    # Adjust for Volume
    volume_adjustment = (df['volume'] / df['volume'].rolling(window=5).mean()) + 1
    
    # Combine Intraday Range and Volume Adjustment
    combined_metric = intraday_range * volume_adjustment
    
    # Smoothing with EMA
    smoothed_metric = combined_metric.ewm(span=5, adjust=False).mean()
    
    # Calculate EMA Cross Signal
    short_ema = df['close'].ewm(span=5, adjust=False).mean()
    long_ema = df['close'].ewm(span=20, adjust=False).mean()
    
    # Determine Momentum
    momentum = (short_ema > long_ema).astype(int) - (short_ema < long_ema).astype(int)
    
    # Calculate Intraday Momentum
    high_low_diff = df['high'] - df['low']
    open_close_return = df['close'] - df['open']
    intraday_momentum = 0.6 * high_low_diff + 0.4 * open_close_return
    
    # Calculate On-Balance Volume (OBV)
    obv = df['volume'].copy()
    obv.iloc[0] = 0
    obv[1:] = np.where(df['close'] > df['close'].shift(1), obv[1:] + df['volume'][1:],
                       np.where(df['close'] < df['close'].shift(1), obv[1:] - df['volume'][1:], obv[1:]))
    
    # Weight OBV by Difference between Short and Long EMA
    weight = (short_ema - long_ema).abs()
    final_obv_factor = obv * weight
    
    # Apply Volume and Amount Shock Filter
    volume_ratio = df['volume'] / df['volume'].shift(1)
    amount = df['close'] * df['volume']
    amount_ratio = amount / amount.shift(1)
    shock_filter = (volume_ratio > 1.5) & (amount_ratio > 1.5)
    
    # Measure Volume Synchronization
    volume_change = df['volume'] - df['volume'].shift(1)
    price_change = df['close'] - df['close'].shift(1)
    synchronized_product = volume_change * price_change
    synchronization_value = np.sign(synchronized_product)
    
    # Synthesize Combined Factor
    combined_factor = smoothed_metric + synchronization_value
    combined_factor = combined_factor.clip(lower=0)
    
    # Final Alpha Factor
    final_alpha_factor = combined_factor * np.where(shock_filter, final_obv_factor, 0)
    
    return final_alpha_factor
