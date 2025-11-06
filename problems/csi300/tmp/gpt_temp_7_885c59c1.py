import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Daily Price Range Efficiency (avoid division by zero)
    data['range_efficiency'] = np.where(data['volume'] > 0, 
                                       data['true_range'] / data['volume'], 
                                       0)
    
    # Calculate Opening Gap
    data['opening_gap'] = abs(data['open'] - data['prev_close'])
    
    # Estimate first hour volume (using rolling 1-hour sum, assuming 1 row per hour)
    # For daily data, we'll use 25% of daily volume as first hour estimate
    data['first_hour_volume_est'] = data['volume'] * 0.25
    
    # Opening Gap Efficiency
    data['gap_efficiency'] = np.where(data['first_hour_volume_est'] > 0,
                                     data['opening_gap'] / data['first_hour_volume_est'],
                                     0)
    
    # Calculate returns for volatility
    data['returns'] = data['close'].pct_change()
    
    # 20-day Rolling Volatility
    data['volatility_20d'] = data['returns'].rolling(window=20, min_periods=10).std()
    
    # Volatility regime classification
    vol_threshold_high = data['volatility_20d'].rolling(window=100, min_periods=50).quantile(0.8)
    vol_threshold_low = data['volatility_20d'].rolling(window=100, min_periods=50).quantile(0.2)
    
    data['vol_regime'] = 'normal'
    data.loc[data['volatility_20d'] > vol_threshold_high, 'vol_regime'] = 'high'
    data.loc[data['volatility_20d'] < vol_threshold_low, 'vol_regime'] = 'low'
    
    # Calculate Price Slope (20-day trend)
    def calculate_slope(series):
        if len(series) < 5:
            return 0
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    data['price_slope'] = data['close'].rolling(window=20, min_periods=10).apply(
        calculate_slope, raw=False
    )
    
    # Trend regime classification
    slope_std = data['price_slope'].rolling(window=100, min_periods=50).std()
    data['trend_regime'] = 'no_trend'
    data.loc[data['price_slope'] > 2 * slope_std, 'trend_regime'] = 'uptrend'
    data.loc[data['price_slope'] < -2 * slope_std, 'trend_regime'] = 'downtrend'
    
    # Volatility regime weights
    vol_weights = {
        'high': 0.5,    # reduce weight by 50%
        'normal': 1.0,  # no adjustment
        'low': 1.3      # increase weight by 30%
    }
    
    # Apply volatility regime adjustment
    data['range_eff_vol_adj'] = data.apply(
        lambda row: row['range_efficiency'] * vol_weights[row['vol_regime']], 
        axis=1
    )
    data['gap_eff_vol_adj'] = data.apply(
        lambda row: row['gap_efficiency'] * vol_weights[row['vol_regime']], 
        axis=1
    )
    
    # Trend-based weighting
    def calculate_trend_weighted_efficiency(row):
        if row['trend_regime'] == 'uptrend':
            return 0.7 * row['gap_eff_vol_adj'] + 0.3 * row['range_eff_vol_adj']
        elif row['trend_regime'] == 'downtrend':
            return 0.3 * row['gap_eff_vol_adj'] + 0.7 * row['range_eff_vol_adj']
        else:  # no trend
            return 0.5 * row['gap_eff_vol_adj'] + 0.5 * row['range_eff_vol_adj']
    
    data['regime_adjusted_efficiency'] = data.apply(calculate_trend_weighted_efficiency, axis=1)
    
    # Generate predictive signal
    def generate_signal(row):
        base_signal = row['regime_adjusted_efficiency']
        
        # High efficiency in low volatility → bullish
        if row['vol_regime'] == 'low' and row['regime_adjusted_efficiency'] > 0:
            signal_strength = 1.5
        # Low efficiency in high volatility → bearish  
        elif row['vol_regime'] == 'high' and row['regime_adjusted_efficiency'] < 0:
            signal_strength = -1.5
        else:
            signal_strength = 1.0
            
        # Trend confirmation
        if row['trend_regime'] == 'uptrend' and base_signal > 0:
            signal_strength *= 1.2
        elif row['trend_regime'] == 'downtrend' and base_signal < 0:
            signal_strength *= 1.2
            
        return base_signal * signal_strength
    
    data['factor'] = data.apply(generate_signal, axis=1)
    
    # Normalize the factor
    factor_mean = data['factor'].rolling(window=100, min_periods=50).mean()
    factor_std = data['factor'].rolling(window=100, min_periods=50).std()
    data['normalized_factor'] = (data['factor'] - factor_mean) / factor_std
    
    return data['normalized_factor']
