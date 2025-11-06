import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-adjusted trend persistence,
    volume-weighted price acceleration, bidirectional gap analysis, and 
    price-volume cointegration.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility-Adjusted Trend Persistence
    # Price Trend Strength
    # 10-day linear regression slope
    def rolling_slope(x):
        if len(x) < 2:
            return np.nan
        return stats.linregress(range(len(x)), x)[0]
    
    data['trend_slope_10d'] = data['close'].rolling(window=10, min_periods=2).apply(rolling_slope, raw=True)
    
    # 20-day price channel position
    data['high_20d'] = data['high'].rolling(window=20, min_periods=1).max()
    data['low_20d'] = data['low'].rolling(window=20, min_periods=1).min()
    data['channel_position'] = (data['close'] - data['low_20d']) / (data['high_20d'] - data['low_20d'] + 1e-8)
    
    # Volatility Normalization
    data['volatility_20d'] = data['close'].pct_change().rolling(window=20, min_periods=1).std()
    data['trend_to_vol_ratio'] = data['trend_slope_10d'] / (data['volatility_20d'] + 1e-8)
    
    # 2. Volume-Weighted Price Acceleration
    # Price Acceleration
    data['return_5d'] = data['close'].pct_change(periods=5)
    data['return_10d'] = data['close'].pct_change(periods=10)
    data['price_acceleration'] = data['return_5d'] - data['return_10d']
    
    # Second derivative of price (acceleration of acceleration)
    data['return_3d'] = data['close'].pct_change(periods=3)
    data['price_accel_2nd'] = data['return_5d'] - 2 * data['return_3d']
    
    # Volume Weighting
    data['vwap'] = (data['close'] * data['volume']).rolling(window=5, min_periods=1).sum() / data['volume'].rolling(window=5, min_periods=1).sum()
    data['vwap_change'] = data['vwap'].pct_change()
    data['volume_weighted_accel'] = data['price_acceleration'] * data['vwap_change']
    
    # 3. Bidirectional Gap Analysis
    # Gap Direction Classification
    data['prev_close'] = data['close'].shift(1)
    data['gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_up'] = np.where(data['gap'] > 0, data['gap'], 0)
    data['gap_down'] = np.where(data['gap'] < 0, -data['gap'], 0)
    
    # Gap magnitude ranking (within rolling window)
    data['gap_magnitude_rank'] = data['gap'].abs().rolling(window=20, min_periods=1).rank(pct=True)
    
    # Post-Gap Behavior
    data['post_gap_return_3d'] = data['close'].shift(-3) / data['close'] - 1
    data['volume_persistence'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    
    # 4. Price-Volume Cointegration
    # Long-term Relationship
    data['log_price'] = np.log(data['close'])
    data['log_volume'] = np.log(data['volume'] + 1e-8)
    
    # 60-day price-volume correlation
    data['price_volume_corr_60d'] = data['log_price'].rolling(window=60, min_periods=1).corr(data['log_volume'])
    
    # Residual from price-volume regression
    def rolling_residual(x):
        if len(x) < 2:
            return np.nan
        y = x[:, 0]  # price
        x_vals = x[:, 1]  # volume
        try:
            slope, intercept, _, _, _ = stats.linregress(x_vals, y)
            residuals = y - (intercept + slope * x_vals)
            return residuals[-1] if len(residuals) > 0 else np.nan
        except:
            return np.nan
    
    price_volume_data = np.column_stack([data['log_price'], data['log_volume']])
    data['price_volume_residual'] = pd.Series(
        [rolling_residual(price_volume_data[max(0, i-59):i+1]) for i in range(len(data))],
        index=data.index
    )
    
    # Short-term Deviation
    data['residual_mean'] = data['price_volume_residual'].rolling(window=20, min_periods=1).mean()
    data['residual_std'] = data['price_volume_residual'].rolling(window=20, min_periods=1).std()
    data['residual_zscore'] = (data['price_volume_residual'] - data['residual_mean']) / (data['residual_std'] + 1e-8)
    
    # Mean reversion speed (negative of z-score indicates reversion tendency)
    data['mean_reversion_speed'] = -data['residual_zscore']
    
    # Combine factors with appropriate weights
    factors = [
        data['trend_to_vol_ratio'],           # Volatility-adjusted trend
        data['volume_weighted_accel'],        # Volume-weighted acceleration
        data['gap_magnitude_rank'] * np.sign(data['gap']),  # Directional gap strength
        data['mean_reversion_speed'],         # Price-volume mean reversion
        data['price_volume_corr_60d']         # Long-term price-volume relationship
    ]
    
    # Normalize each factor and combine
    alpha_factor = pd.Series(0, index=data.index)
    weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Weights based on factor importance
    
    for i, factor in enumerate(factors):
        # Z-score normalization within rolling window
        factor_normalized = (factor - factor.rolling(window=60, min_periods=1).mean()) / (factor.rolling(window=60, min_periods=1).std() + 1e-8)
        alpha_factor += weights[i] * factor_normalized
    
    # Remove any forward-looking data references
    data = data.drop(columns=['post_gap_return_3d'])
    
    return alpha_factor
