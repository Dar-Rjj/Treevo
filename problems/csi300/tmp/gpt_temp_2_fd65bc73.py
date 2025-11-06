import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Compute Momentum Acceleration
    # Calculate Price Momentum
    data['price_change_3d'] = data['close'] / data['close'].shift(2) - 1
    data['price_change_6d'] = data['close'] / data['close'].shift(5) - 1
    
    # Calculate Volume Momentum
    data['volume_change_3d'] = data['volume'] / data['volume'].shift(2) - 1
    data['volume_change_6d'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Compute Convergence Signals
    # Calculate Momentum Alignment
    data['momentum_align_3d'] = data['price_change_3d'] * data['volume_change_3d']
    data['momentum_align_6d'] = data['price_change_6d'] * data['volume_change_6d']
    
    # Compute Acceleration Divergence
    data['accel_divergence'] = data['momentum_align_3d'] / data['momentum_align_6d']
    data['accel_divergence'] = np.tanh(data['accel_divergence'])
    
    # Incorporate Price Efficiency
    # Calculate Intraday Efficiency
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['price_efficiency'] = data['price_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate 3-day Efficiency Trend
    def calc_efficiency_trend(series):
        if len(series) < 3 or series.isna().any():
            return np.nan
        x = np.arange(3)
        slope, _, _, _, _ = linregress(x, series.values)
        return slope
    
    data['efficiency_trend'] = data['price_efficiency'].rolling(window=3, min_periods=3).apply(
        calc_efficiency_trend, raw=False
    )
    
    # Compute Volume Concentration
    data['avg_trade_size'] = data['amount'] / data['volume']
    data['avg_trade_size'] = data['avg_trade_size'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate 3-day Trade Size Trend
    data['trade_size_trend'] = data['avg_trade_size'].rolling(window=3, min_periods=3).apply(
        calc_efficiency_trend, raw=False
    )
    
    # Generate Composite Factor
    # Combine Convergence and Efficiency
    data['composite_factor'] = data['accel_divergence'] * data['efficiency_trend'] * data['trade_size_trend']
    
    # Apply Dynamic Scaling
    # Compute Recent Volatility (std dev of High-Low from t-4 to t)
    data['high_low_range'] = data['high'] - data['low']
    data['recent_volatility'] = data['high_low_range'].rolling(window=5, min_periods=5).std()
    
    # Scale Factor by Volatility Reciprocal
    data['scaled_factor'] = data['composite_factor'] / data['recent_volatility']
    data['scaled_factor'] = data['scaled_factor'].replace([np.inf, -np.inf], np.nan)
    
    # Final Factor Adjustment
    # Apply cubic root transformation and preserve original sign
    sign_preserver = np.sign(data['accel_divergence'])
    data['final_factor'] = sign_preserver * np.cbrt(np.abs(data['scaled_factor']))
    
    return data['final_factor']
