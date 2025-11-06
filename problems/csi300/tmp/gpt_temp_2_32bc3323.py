import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adaptive Gap Efficiency Momentum factor
    Combines gap momentum, volatility scaling, and price efficiency for predictive signal
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # Gap Momentum Analysis
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['intraday_range'] = (data['high'] - data['low']) / data['open']
    data['gap_efficiency'] = np.where(
        data['overnight_gap'] != 0,
        data['intraday_range'] / abs(data['overnight_gap']),
        0
    )
    
    # Volatility Scaling
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    data['avg_true_range'] = data['true_range'].rolling(window=10, min_periods=5).mean()
    data['volatility_scale'] = data['avg_true_range'].rolling(window=5, min_periods=3).apply(
        lambda x: 1.0 if x.iloc[-1] > x.median() else 0.5
    )
    
    # Daily Price Efficiency Ratio
    data['close_open_range'] = abs(data['close'] - data['open']) / data['open']
    data['efficiency_ratio'] = np.where(
        data['intraday_range'] > 0,
        data['close_open_range'] / data['intraday_range'],
        0
    )
    
    # Efficiency-Weighted Signal
    data['scaled_gap_momentum'] = data['overnight_gap'] * data['volatility_scale']
    data['efficiency_trend'] = data['efficiency_ratio'].rolling(window=5, min_periods=3).mean()
    
    # Final factor calculation
    factor = data['scaled_gap_momentum'] * data['efficiency_trend'] * data['gap_efficiency']
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return factor
