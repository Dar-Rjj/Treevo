import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Price-Volume Divergence with Regime Switching alpha factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price Momentum components
    data['price_return_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_acceleration'] = data['price_return_5d'] - data['price_return_5d'].shift(5)
    
    # Calculate Volume Anomaly components
    data['volume_mean_20d'] = data['volume'].rolling(window=20).mean()
    data['volume_std_20d'] = data['volume'].rolling(window=20).std()
    data['volume_zscore'] = (data['volume'] - data['volume_mean_20d']) / data['volume_std_20d']
    
    # Calculate Volume Trend using linear regression slope
    def volume_slope(volume_series):
        if len(volume_series) < 5:
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = stats.linregress(x, volume_series)
        return slope
    
    data['volume_trend'] = data['volume'].rolling(window=5).apply(volume_slope, raw=True)
    
    # Calculate Volume Volatility
    data['volume_max_10d'] = data['volume'].rolling(window=10).max()
    data['volume_min_10d'] = data['volume'].rolling(window=10).min()
    data['volume_range_ratio'] = (data['volume_max_10d'] - data['volume_min_10d']) / data['volume_mean_20d']
    
    # Calculate Hurst Exponent for market regime detection
    def hurst_exponent(ts):
        if len(ts) < 20:
            return np.nan
        lags = range(2, 10)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    data['hurst_20d'] = data['close'].rolling(window=20).apply(hurst_exponent, raw=True)
    
    # Calculate Volatility Persistence
    data['abs_return_5d'] = data['price_return_5d'].abs()
    data['vol_persistence'] = data['abs_return_5d'].rolling(window=20).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) == 20 else np.nan, raw=False
    )
    
    # Define regime probabilities using fuzzy logic
    # Trending vs Mean-reverting regime
    data['trend_prob'] = 1 / (1 + np.exp(-10 * (data['hurst_20d'] - 0.5)))
    data['mean_rev_prob'] = 1 - data['trend_prob']
    
    # Volatile vs Calm regime
    data['volatile_prob'] = 1 / (1 + np.exp(-5 * (data['vol_persistence'] - 0.3)))
    data['calm_prob'] = 1 - data['volatile_prob']
    
    # Calculate regime quadrant probabilities
    data['prob_trend_vol'] = data['trend_prob'] * data['volatile_prob']
    data['prob_trend_calm'] = data['trend_prob'] * data['calm_prob']
    data['prob_meanrev_vol'] = data['mean_rev_prob'] * data['volatile_prob']
    data['prob_meanrev_calm'] = data['mean_rev_prob'] * data['calm_prob']
    
    # Normalize probabilities to sum to 1
    total_prob = (data['prob_trend_vol'] + data['prob_trend_calm'] + 
                  data['prob_meanrev_vol'] + data['prob_meanrev_calm'])
    data['prob_trend_vol'] /= total_prob
    data['prob_trend_calm'] /= total_prob
    data['prob_meanrev_vol'] /= total_prob
    data['prob_meanrev_calm'] /= total_prob
    
    # Calculate regime-specific factors
    # Trending Volatile Regime
    data['factor_tv'] = np.tanh(data['price_acceleration'] * data['volume_zscore'])
    # Apply exponential decay for shorter persistence
    data['factor_tv'] = data['factor_tv'] * np.exp(-0.1 * np.arange(len(data)))
    
    # Trending Calm Regime
    data['factor_tc'] = (data['price_acceleration'] ** 2) * data['volume_trend']
    # Apply convex transformation
    data['factor_tc'] = np.sign(data['factor_tc']) * (np.abs(data['factor_tc']) ** 0.7)
    
    # Mean-reverting Volatile Regime
    data['price_ma_10d'] = data['close'].rolling(window=10).mean()
    data['distance_from_ma'] = (data['close'] - data['price_ma_10d']) / data['price_ma_10d']
    data['factor_mv'] = (-data['price_return_5d'] * data['volume_range_ratio'] * 
                         data['distance_from_ma'] / (data['abs_return_5d'] + 1e-8))
    
    # Mean-reverting Calm Regime
    data['price_range_10d'] = (data['high'].rolling(window=10).max() - 
                              data['low'].rolling(window=10).min()) / data['close']
    data['price_position'] = (data['close'] - data['low'].rolling(window=10).min()) / (
        data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min() + 1e-8)
    # Contrarian volume signal
    data['factor_mc'] = (-data['volume_zscore'] * (data['price_position'] - 0.5) * 
                         data['volume_trend'] / (data['price_range_10d'] + 1e-8))
    
    # Combine regime factors with probabilities
    data['combined_factor'] = (
        data['prob_trend_vol'] * data['factor_tv'] +
        data['prob_trend_calm'] * data['factor_tc'] +
        data['prob_meanrev_vol'] * data['factor_mv'] +
        data['prob_meanrev_calm'] * data['factor_mc']
    )
    
    # Apply regime transition smoothing
    data['smoothed_factor'] = data['combined_factor'].ewm(span=5).mean()
    
    # Risk-adjusted scaling using rolling volatility
    data['factor_volatility'] = data['smoothed_factor'].rolling(window=20).std()
    data['final_alpha'] = data['smoothed_factor'] / (data['factor_volatility'] + 1e-8)
    
    # Normalize final output
    data['final_alpha'] = (data['final_alpha'] - data['final_alpha'].rolling(window=50).mean()) / (
        data['final_alpha'].rolling(window=50).std() + 1e-8)
    
    return data['final_alpha']
