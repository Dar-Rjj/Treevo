import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    High-Frequency Reversal with Liquidity Acceleration factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Reversal Component
    # Intraday Reversal Signal
    data['daily_range'] = data['high'] - data['low']
    data['price_change'] = (data['close'] - data['open']).abs()
    data['range_efficiency'] = data['daily_range'] / (data['price_change'] + 1e-8)
    
    # Gap-Fill Behavior
    data['overnight_gap'] = data['open'] - data['close'].shift(1)
    data['intraday_fill'] = data['close'] - data['open']
    
    # Short-Term Overreaction
    data['return_1d'] = data['close'] / data['close'].shift(1) - 1
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    data['price_acceleration'] = data['return_1d'] - data['return_3d']
    
    # Price Exhaustion
    # Consecutive same-direction closes
    data['price_direction'] = np.sign(data['close'] - data['close'].shift(1))
    data['consecutive_direction'] = 0
    for i in range(1, 6):
        data['consecutive_direction'] += (data['price_direction'] == data['price_direction'].shift(i)).astype(int)
    
    # True Range calculations
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = (data['high'] - data['close'].shift(1)).abs()
    data['tr3'] = (data['low'] - data['close'].shift(1)).abs()
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_5d'] = data['true_range'].rolling(window=5).mean()
    data['atr_expansion'] = data['true_range'] / (data['atr_5d'] + 1e-8)
    
    # Liquidity Acceleration Component
    # Volume Velocity
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_momentum'] = data['volume'] / (data['volume_5d_avg'] + 1e-8)
    
    # Volume Breakout
    data['volume_20d_mean'] = data['volume'].rolling(window=20).mean()
    data['volume_20d_std'] = data['volume'].rolling(window=20).std()
    data['volume_zscore'] = (data['volume'] - data['volume_20d_mean']) / (data['volume_20d_std'] + 1e-8)
    
    # Liquidity Quality
    data['amount_per_volume'] = data['amount'] / (data['volume'] + 1e-8)
    
    # Liquidity Persistence
    data['liquidity_median'] = data['amount_per_volume'].rolling(window=10).median()
    data['above_median_liquidity'] = (data['amount_per_volume'] > data['liquidity_median']).astype(int)
    data['liquidity_persistence'] = data['above_median_liquidity'].rolling(window=10).sum()
    
    # Liquidity trend using linear regression
    def liquidity_trend(x):
        if len(x) < 3:
            return 0
        try:
            slope, _, _, _, _ = linregress(range(len(x)), x)
            return slope
        except:
            return 0
    
    data['liquidity_trend'] = data['amount_per_volume'].rolling(window=10).apply(liquidity_trend, raw=True)
    
    # Factor Synthesis
    # Reversal Core
    reversal_core = (
        data['range_efficiency'] * 
        np.sign(data['intraday_fill']) * 
        data['price_acceleration'] * 
        (1 + data['consecutive_direction'] / 5) * 
        (1 + data['atr_expansion'])
    )
    
    # Liquidity Filters
    liquidity_filter = (
        data['volume_momentum'] * 
        (1 + data['volume_zscore'].clip(-3, 3) / 3) * 
        data['amount_per_volume'] * 
        (1 + data['liquidity_persistence'] / 10) * 
        (1 + data['liquidity_trend'])
    )
    
    # Final Alpha Factor
    alpha_factor = reversal_core * liquidity_filter
    
    # Clean and normalize
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = alpha_factor.fillna(0)
    
    # Remove any lookahead bias by ensuring no future data
    return alpha_factor
