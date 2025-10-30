import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns and ranges
    data['daily_return'] = data['close'] / data['open'] - 1
    data['daily_range'] = (data['high'] - data['low']) / data['open']
    
    # Multi-Timeframe Overnight Momentum
    data['overnight_return'] = data['close'].shift(1) / data['open'] - 1
    data['overnight_3d'] = data['overnight_return'].rolling(window=3, min_periods=3).sum()
    data['overnight_5d'] = data['overnight_return'].rolling(window=5, min_periods=5).sum()
    
    # Multi-Timeframe Intraday Momentum
    data['intraday_3d'] = data['daily_return'].rolling(window=3, min_periods=3).sum()
    data['intraday_5d'] = data['daily_return'].rolling(window=5, min_periods=5).sum()
    
    # Acceleration Divergence
    data['accel_short'] = data['intraday_3d'] - data['overnight_3d']
    data['accel_medium'] = data['intraday_5d'] - data['overnight_5d']
    data['accel_divergence'] = data['accel_short'] - data['accel_medium']
    
    # Multi-Timeframe Volume Analysis
    data['volume_3d_avg'] = data['volume'].rolling(window=3, min_periods=3).mean()
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=5).mean()
    data['volume_divergence'] = (data['volume_3d_avg'] / data['volume_5d_avg']) - 1
    
    # Price-Volume Consistency
    data['price_volume_corr_3d'] = data['daily_return'].rolling(window=3, min_periods=3).corr(data['volume'])
    data['price_volume_corr_5d'] = data['daily_return'].rolling(window=5, min_periods=5).corr(data['volume'])
    data['corr_divergence'] = data['price_volume_corr_3d'] - data['price_volume_corr_5d']
    
    # Volume Efficiency Signal
    data['volume_efficiency'] = data['volume_divergence'] * data['corr_divergence']
    data['volume_efficiency_aligned'] = data['volume_efficiency'] * np.sign(data['accel_divergence'])
    
    # Momentum Persistence Weighting
    data['accel_sign'] = np.sign(data['accel_divergence'])
    data['sign_consistency_3d'] = data['accel_sign'].rolling(window=3, min_periods=3).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) == 3 else np.nan
    )
    data['consistency_ratio'] = data['sign_consistency_3d'] / 3
    
    # Core Signal
    data['core_signal'] = data['accel_divergence'] * data['volume_efficiency_aligned']
    data['weighted_core_signal'] = data['core_signal'] * data['consistency_ratio']
    
    # Multi-Timeframe Volatility Adjustment
    data['volatility_3d'] = data['daily_range'].rolling(window=3, min_periods=3).std()
    data['volatility_5d'] = data['daily_range'].rolling(window=5, min_periods=5).std()
    data['volatility_ratio'] = data['volatility_3d'] / (data['volatility_5d'] + 1e-8)
    data['volatility_weighted_signal'] = data['weighted_core_signal'] / (data['volatility_ratio'] + 1e-8)
    
    # Multi-Timeframe Trend Confirmation
    def linear_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]
    
    data['trend_3d'] = data['close'].rolling(window=3, min_periods=3).apply(linear_slope, raw=False)
    data['trend_5d'] = data['close'].rolling(window=5, min_periods=5).apply(linear_slope, raw=False)
    data['trend_divergence'] = data['trend_3d'] - data['trend_5d']
    
    # Final Alpha Generation
    data['alpha_factor'] = data['volatility_weighted_signal'] * data['trend_divergence']
    
    # Apply bounded output constraints
    data['alpha_factor'] = np.tanh(data['alpha_factor'] / (data['alpha_factor'].std() + 1e-8))
    
    return data['alpha_factor']
