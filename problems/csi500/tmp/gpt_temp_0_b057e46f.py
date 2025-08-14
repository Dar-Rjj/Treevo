import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from arch import arch_model

def heuristics_v2(df):
    # Compute Daily Return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate Volume-Weighted Daily Return
    df['volume_weighted_return'] = df['daily_return'] * df['volume']
    
    # Calculate Exponential Weighted Moving Average (EWMA) of Volume-Weighted Returns
    decay_factor = 0.94
    df['ewma_vol_weighted_returns'] = df['volume_weighted_return'].ewm(alpha=1 - decay_factor).mean()
    
    # Calculate Typical Price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate Volume-Weighted Typical Price
    df['vwap'] = (df['volume'] * df['typical_price']).cumsum() / df['volume'].cumsum()
    
    # Calculate Long-Term Dynamic Trend
    df['50_day_ma'] = df['close'].rolling(window=50).mean()
    df['200_day_ma'] = df['close'].rolling(window=200).mean()
    df['long_term_trend'] = df['50_day_ma'] - df['200_day_ma']
    
    # Calculate Adaptive Volatility Index using GARCH(1,1)
    returns = df['close'].pct_change().dropna()
    garch_model = arch_model(returns, vol='Garch', p=1, q=1)
    garch_res = garch_model.fit(disp='off')
    df['garch_vol'] = garch_res.conditional_volatility.reindex(df.index).fillna(method='ffill')
    
    # Calculate Trend Strength
    df['trend_strength'] = df['50_day_ma'] / df['200_day_ma']
    
    # Calculate Liquidity Measure
    df['vwap_ratio'] = df['vwap'] / df['close']
    
    # Final Factor Calculation
    df['alpha_factor'] = (df['ewma_vol_weighted_returns'] * df['vwap']) + \
                         df['long_term_trend'] - \
                         df['garch_vol'] + \
                         df['trend_strength'] * \
                         df['vwap_ratio']
    
    return df['alpha_factor']
