import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Volatility-Adjusted Momentum
    # Short-Term Momentum Calculation
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Volatility Estimation
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['avg_range_5d'] = data['daily_range'].rolling(window=5).mean()
    
    # Volatility Adjustment
    data['vol_adj_mom_5d'] = data['mom_5d'] / data['avg_range_5d']
    data['vol_adj_mom_10d'] = data['mom_10d'] / data['avg_range_5d']
    
    # Volume Divergence Analysis
    # Volume Trend Calculation
    def calc_volume_slope(volume_series):
        if len(volume_series) < 5:
            return np.nan
        x = np.arange(5)
        slope, _, _, _, _ = linregress(x, volume_series)
        return slope
    
    data['volume_slope'] = data['volume'].rolling(window=5).apply(calc_volume_slope, raw=True)
    data['volume_direction'] = np.sign(data['volume_slope'])
    
    # Price Momentum Direction
    data['mom_5d_direction'] = np.sign(data['mom_5d'])
    data['mom_10d_direction'] = np.sign(data['mom_10d'])
    
    # Divergence Detection
    conditions = [
        (data['mom_5d_direction'] < 0) & (data['volume_direction'] > 0),  # Positive divergence
        (data['mom_5d_direction'] > 0) & (data['volume_direction'] < 0),  # Negative divergence
        (data['mom_5d_direction'] * data['volume_direction'] > 0)         # Confirmation
    ]
    choices = [1.4, 0.6, 1.0]  # Multipliers
    data['volume_multiplier'] = np.select(conditions, choices, default=1.0)
    
    # Volatility Regime Classification
    data['median_range_20d'] = data['daily_range'].rolling(window=20).median()
    
    # Regime Assignment
    regime_conditions = [
        data['daily_range'] > (1.5 * data['median_range_20d']),  # High volatility
        data['daily_range'] < (0.7 * data['median_range_20d']),  # Low volatility
    ]
    regime_choices = [0.8, 1.2]  # Multipliers
    data['regime_multiplier'] = np.select(regime_conditions, regime_choices, default=1.0)
    
    # Alpha Signal Generation
    # Base Signal Construction
    data['base_signal'] = (data['vol_adj_mom_5d'] + data['vol_adj_mom_10d']) / 2
    
    # Final Alpha Signal
    data['alpha_signal'] = data['base_signal'] * data['volume_multiplier'] * data['regime_multiplier']
    
    return data['alpha_signal']
