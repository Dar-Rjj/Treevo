import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price Momentum
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Compute Range Efficiency
    # True Range calculation
    prev_close = data['close'].shift(1)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - prev_close)
    tr3 = abs(data['low'] - prev_close)
    data['true_range'] = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Price movement
    data['price_movement'] = abs(data['close'] - prev_close)
    
    # Efficiency Ratio over 5-day window
    data['efficiency_ratio'] = data['price_movement'].rolling(window=5).sum() / data['true_range'].rolling(window=5).sum()
    
    # Identify Breakout Conditions
    # Volume Breakout Detection
    data['volume_sma_20d'] = data['volume'].rolling(window=20).mean()
    data['volume_breakout'] = (data['volume'] > 1.5 * data['volume_sma_20d']).astype(int)
    
    # Range Breakout Detection
    data['daily_range'] = (data['high'] - data['low']) / prev_close
    data['avg_range_20d'] = data['daily_range'].rolling(window=20).mean()
    data['range_breakout'] = (data['daily_range'] > 1.5 * data['avg_range_20d']).astype(int)
    
    # Combine Momentum and Efficiency
    data['momentum_efficiency'] = data['momentum_5d'] * data['efficiency_ratio']
    data['breakout_filter'] = (data['volume_breakout'] == 1) & (data['range_breakout'] == 1)
    data['filtered_momentum'] = data['momentum_efficiency'] * data['breakout_filter'] * data['momentum_10d']
    
    # Volume-Weighted Signal Integration
    # Volume Acceleration Component
    data['volume_roc_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_acceleration'] = data['volume_roc_5d'] - data['volume_roc_5d'].shift(1)
    
    # Liquidity Adjustment
    data['amount_percentile'] = data['amount'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    data['liquidity_factor'] = 0.5 + 0.5 * data['amount_percentile']
    
    # Final Alpha Generation
    data['raw_alpha'] = data['filtered_momentum'] * data['volume_acceleration'] * data['liquidity_factor']
    data['alpha_factor'] = data['raw_alpha'].ewm(span=3, adjust=False).mean()
    
    # Clean up and return
    alpha_series = data['alpha_factor'].copy()
    return alpha_series
