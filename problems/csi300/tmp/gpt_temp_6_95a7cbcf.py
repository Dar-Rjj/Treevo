import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Volatility Components
    # Intraday Volatility
    data['intraday_vol'] = (data['high'] - data['low']) / data['close']
    
    # Gap Volatility
    data['gap_vol'] = np.abs(data['open'] / data['close'].shift(1) - 1)
    
    # Trend Volatility (using rolling window)
    data['trend_vol'] = data['close'].rolling(window=5).std() / data['close'].rolling(window=5).mean()
    
    # Volume Components
    # Volume Concentration
    data['vol_concentration'] = data['volume'] / data['volume'].rolling(window=5).sum()
    
    # Volume Persistence
    vol_diff = data['volume'].diff()
    data['vol_persistence'] = (
        vol_diff.rolling(window=5).apply(lambda x: (x > 0).sum() - (x < 0).sum(), raw=False)
    )
    
    # Volume-Volatility Divergence
    vol_change = data['volume'] / data['volume'].shift(1) - 1
    intraday_vol_change = data['intraday_vol'] / data['intraday_vol'].shift(1) - 1
    data['vol_vol_divergence'] = vol_change - intraday_vol_change
    
    # Interaction Terms
    # Volatility Efficiency
    price_change = data['close'] - data['close'].shift(1)
    high_low_range = data['high'] - data['low']
    # Avoid division by zero
    high_low_range = high_low_range.replace(0, np.nan)
    data['vol_efficiency'] = (price_change / high_low_range) * data['volume']
    
    # Gap Absorption
    close_open_diff = data['close'] - data['open']
    open_prev_close_diff = data['open'] - data['close'].shift(1)
    # Avoid division by zero
    open_prev_close_diff = open_prev_close_diff.replace(0, np.nan)
    data['gap_absorption'] = (close_open_diff / open_prev_close_diff) * data['volume']
    
    # Alpha Construction
    # Volatility-Adjusted Volume
    data['vol_adj_volume'] = (
        data['vol_concentration'] * 
        data['vol_efficiency'] * 
        data['vol_persistence']
    )
    
    # Final Alpha
    alpha = (
        data['vol_adj_volume'] * 
        data['gap_absorption'] * 
        (1 + data['vol_vol_divergence'])
    )
    
    return alpha
