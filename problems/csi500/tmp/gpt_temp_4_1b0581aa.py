import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Reversal with Volume Confirmation factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Short-Term Reversal Component
    # Recent price returns
    data['ret_1d'] = data['close'].pct_change(1)
    data['ret_5d'] = data['close'].pct_change(5)
    
    # Acceleration factor - second derivative of price
    data['price_diff_1'] = data['close'] - data['close'].shift(1)
    data['price_diff_2'] = data['close'].shift(1) - data['close'].shift(2)
    data['acceleration'] = data['price_diff_1'] - data['price_diff_2']
    
    # Volatility adjustment
    data['range_current'] = data['high'] - data['low']
    data['range_prev'] = data['high'].shift(1) - data['low'].shift(1)
    data['vol_ratio'] = data['range_current'] / (data['range_prev'] + 1e-8)
    
    # Volatility-weighted acceleration
    data['vol_weighted_accel'] = data['acceleration'] * data['vol_ratio']
    
    # Reversal factor combining short-term returns and acceleration
    data['reversal_factor'] = -data['ret_1d'] * data['vol_weighted_accel']
    
    # Volume Confirmation Component
    # Volume percentile (20-day lookback)
    data['volume_percentile'] = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=True
    )
    
    # Volume spike detection
    data['volume_median_10d'] = data['volume'].rolling(window=10, min_periods=5).median()
    data['volume_ratio'] = data['volume'] / (data['volume_median_10d'] + 1e-8)
    
    # Directional volume strength
    data['volume_strength'] = np.where(
        data['ret_1d'] * data['volume_ratio'] > 0,
        data['volume_ratio'],
        -data['volume_ratio']
    )
    
    # Volume confirmation signal
    data['volume_confirmation'] = data['volume_percentile'] * data['volume_strength']
    
    # Combine reversal and volume signals
    data['raw_factor'] = data['reversal_factor'] * data['volume_confirmation']
    
    # Non-linear transformation with hyperbolic tangent
    data['scaled_factor'] = np.tanh(data['raw_factor'] * 10)  # Scale for better sensitivity
    
    # Market regime adjustment using volatility proxy
    data['market_vol_proxy'] = data['range_current'].rolling(window=20, min_periods=10).mean()
    data['vol_regime'] = data['market_vol_proxy'] / data['market_vol_proxy'].rolling(window=60, min_periods=30).median()
    
    # Adjust factor sensitivity based on market regime
    data['regime_adjusted'] = data['scaled_factor'] / (1 + np.abs(data['vol_regime'] - 1))
    
    # Risk Control Layer
    # Calculate factor z-score for winsorization
    factor_mean = data['regime_adjusted'].rolling(window=60, min_periods=30).mean()
    factor_std = data['regime_adjusted'].rolling(window=60, min_periods=30).std()
    data['factor_zscore'] = (data['regime_adjusted'] - factor_mean) / (factor_std + 1e-8)
    
    # Winsorize extreme values at 95th percentile
    data['final_factor'] = np.where(
        data['factor_zscore'].abs() > 1.96,  # ~95th percentile
        np.sign(data['factor_zscore']) * 1.96 * factor_std + factor_mean,
        data['regime_adjusted']
    )
    
    # Stationarity check and adjustment
    autocorr = data['final_factor'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Apply mean reversion adjustment if autocorrelation is too high
    data['stationary_factor'] = np.where(
        autocorr.abs() > 0.3,
        data['final_factor'] - 0.5 * autocorr * data['final_factor'].shift(1),
        data['final_factor']
    )
    
    # Fill NaN values with 0
    result = data['stationary_factor'].fillna(0)
    
    return result
