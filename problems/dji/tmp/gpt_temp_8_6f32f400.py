import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Momentum
    high_low_diff = df['high'] - df['low']
    open_close_return = (df['close'] - df['open']) / df['open']
    
    # Compute Historical Volatility
    df['volatility'] = df[['high', 'low', 'close']].rolling(window=20).std().mean(axis=1)
    
    # Assign Weights based on Volatility
    weight_high_low = 1 / (df['volatility'] + 1e-6)
    weight_open_close = 1 / (df['volatility'] + 1e-6)
    
    # Combine High-Low and Open-Close Returns
    intraday_momentum = (weight_high_low * high_low_diff + weight_open_close * open_close_return) / (weight_high_low + weight_open_close)
    
    # Introduce Non-Linear Transformation
    intraday_momentum = np.exp(intraday_momentum**2)
    
    # Apply Enhanced Volume, Amount, and Price Shock Filter
    volume_ratio = df['volume'] / df['volume'].shift(1)
    amount_ratio = df['amount'] / df['amount'].shift(1)
    price_shock = np.abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # Threshold Filter
    filter_condition = (volume_ratio > 1.5) & (amount_ratio > 1.5) & (price_shock > 0.05)
    intraday_momentum[~filter_condition] = 0
    
    # Introduce Smoothing Factor
    volume_ema = df['volume'].ewm(alpha=0.2, adjust=False).mean()
    amount_ema = df['amount'].ewm(alpha=0.2, adjust=False).mean()
    volume_sma = df['volume'].rolling(window=5).mean()
    amount_sma = df['amount'].rolling(window=5).mean()
    
    # Adjust for Volume
    intraday_momentum /= df['volume']
    
    # Measure Volume Synchronization
    volume_change = df['volume'] - df['volume'].shift(1)
    price_change = df['close'] - df['close'].shift(1)
    synchronized_volume = np.sign(volume_change * price_change)
    
    # Calculate Exponential Moving Average (EMA) Cross Signal
    ema_short = df['close'].ewm(span=5, adjust=False).mean()
    ema_long = df['close'].ewm(span=20, adjust=False).mean()
    
    # Determine Momentum
    momentum = (ema_short > ema_long).astype(int) * 2 - 1
    
    # Calculate On-Balance Volume (OBV)
    obv = df['volume'].copy()
    obv[1:] = np.where(df['close'].diff() > 0, obv.shift(1) + df['volume'],
                       np.where(df['close'].diff() < 0, obv.shift(1) - df['volume'], obv.shift(1)))
    obv.iloc[0] = 0
    
    # Combine Momentum and OBV into a Factor
    momentum_obv_factor = obv * np.abs(ema_short - ema_long)
    
    # Synthesize Final Alpha Factor
    final_alpha_factor = (intraday_momentum + synchronized_volume + momentum_obv_factor)
    
    return final_alpha_factor
