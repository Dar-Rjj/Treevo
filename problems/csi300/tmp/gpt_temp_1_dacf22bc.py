import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Component
    # Intraday Volatility Ratio
    data['intraday_range'] = (data['high'] - data['low']) / data['close']
    data['open_close_gap'] = abs(data['close'] - data['open']) / data['close']
    data['volatility_ratio'] = data['intraday_range'] / (data['open_close_gap'] + 1e-8)
    
    # Volatility Persistence (autocorrelation over 8 days)
    def calc_vol_persistence(series):
        if len(series) < 8:
            return np.nan
        return series.autocorr(lag=1)
    
    data['vol_persistence'] = data['volatility_ratio'].rolling(window=8, min_periods=8).apply(
        calc_vol_persistence, raw=False
    )
    
    # Liquidity Efficiency Component
    # Volume-to-Amount Efficiency
    data['volume_amount_efficiency'] = data['volume'] / (data['amount'] + 1e-8)
    
    # Liquidity Momentum (linear regression slope over 6 days)
    def calc_efficiency_slope(series):
        if len(series) < 6:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return slope
    
    data['liquidity_momentum'] = data['volume_amount_efficiency'].rolling(
        window=6, min_periods=6
    ).apply(calc_efficiency_slope, raw=False)
    
    # Combine components to create final factor
    # Higher volatility persistence and positive liquidity momentum are favorable
    data['factor'] = data['vol_persistence'] * data['liquidity_momentum']
    
    return data['factor']
