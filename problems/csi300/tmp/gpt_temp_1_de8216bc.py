import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday High-Low Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Open-to-Close Change
    open_to_close_change = df['close'] - df['open']
    
    # Compute Weighted Intraday Momentum
    weighted_intraday_momentum = (intraday_range * open_to_close_change) * df['volume']
    
    # Determine Volume Reversal Signal
    daily_volume_change = df['volume'] - df['volume'].shift(1)
    inverse_volume_change = 1 / (daily_volume_change + 1e-6)  # Adding a small value to avoid division by zero
    
    # Apply Exponential Moving Average (EMA) Smoothing
    ema_weighted_intraday_momentum_5 = weighted_intraday_momentum.ewm(span=5, adjust=False).mean()
    ema_weighted_intraday_momentum_20 = weighted_intraday_momentum.ewm(span=20, adjust=False).mean()
    
    ema_inverse_volume_change_5 = inverse_volume_change.ewm(span=5, adjust=False).mean()
    ema_inverse_volume_change_20 = inverse_volume_change.ewm(span=20, adjust=False).mean()
    
    # Integrate Market Context
    market_return = df['market_index_close'].pct_change()
    stock_returns = df['close'].pct_change()
    stock_market_correlation = stock_returns.rolling(window=20).corr(market_return)
    
    # Final Alpha Factor
    final_alpha_factor = (ema_weighted_intraday_momentum_5 * ema_inverse_volume_change_5 + 
                          ema_weighted_intraday_momentum_20 * ema_inverse_volume_change_20) / 2
    
    # Adjust for Market Context
    final_alpha_factor = final_alpha_factor * stock_market_correlation
    
    return final_alpha_factor
