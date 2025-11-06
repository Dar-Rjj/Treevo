import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price Acceleration Analysis
    data['short_term_accel'] = (data['close'] / data['close'].shift(3) - 1) - (data['close'].shift(3) / data['close'].shift(6) - 1)
    data['medium_term_accel'] = (data['close'] / data['close'].shift(5) - 1) - (data['close'].shift(5) / data['close'].shift(10) - 1)
    data['accel_divergence'] = data['short_term_accel'] - data['medium_term_accel']
    
    # Volatility Regime Detection
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Calculate 20-day percentiles for volatility regime
    data['vol_percentile'] = data['true_range'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 80)) * 2 + (x.iloc[-1] < np.percentile(x, 20)) * 0 + 
                 ((x.iloc[-1] >= np.percentile(x, 20)) & (x.iloc[-1] <= np.percentile(x, 80))) * 1,
        raw=False
    )
    
    data['vol_clustering'] = data['true_range'] / data['true_range'].shift(1)
    data['vol_clustering'] = data['vol_clustering'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Microstructure Assessment
    data['spread_proxy'] = 2 * np.abs(data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'])
    data['spread_proxy'] = data['spread_proxy'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['price_discreteness'] = (np.mod(data['close'] * 100, 1)) / data['close']
    data['price_discreteness'] = data['price_discreteness'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['gap_efficiency'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['gap_efficiency'] = data['gap_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volume Analysis
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum'] = data['volume_momentum'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    data['volume_accel'] = (data['volume'] / data['volume'].shift(5) - 1) - (data['volume'].shift(5) / data['volume'].shift(10) - 1)
    data['volume_accel'] = data['volume_accel'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volume-volatility alignment
    data['vol_vol_alignment'] = np.sign(data['volume_momentum']) == np.sign(data['true_range'] / data['true_range'].shift(1) - 1)
    
    # Signal Construction
    # Regime-adaptive base
    data['base_signal'] = np.where(
        data['vol_percentile'] == 2,  # High volatility
        data['accel_divergence'] * data['vol_clustering'],
        np.where(
            data['vol_percentile'] == 0,  # Low volatility
            data['accel_divergence'] / (1 + data['price_discreteness']),
            data['accel_divergence'] * data['volume_momentum']  # Normal volatility
        )
    )
    
    # Microstructure filtering
    data['filtered_signal'] = data['base_signal'] / (1 + data['spread_proxy']) * (1 - data['gap_efficiency'])
    
    # Volume enhancement
    data['volume_enhancement'] = np.where(
        data['vol_vol_alignment'],
        data['filtered_signal'] * data['volume_accel'],  # Confirmation
        data['filtered_signal'] * -1  # Divergence
    )
    
    # Final Alpha Factor
    data['base_condition'] = (data['accel_divergence'] > 0).astype(int)
    data['final_alpha'] = data['base_condition'] * data['filtered_signal'] * data['volume_enhancement']
    
    return data['final_alpha']
