import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Divergence with Momentum Acceleration factor
    """
    data = df.copy()
    
    # Calculate Price-Volume Divergence Metrics
    # Short-Term Divergence (3-day)
    data['price_accel'] = (data['close'] - data['close'].shift(1)) - (data['close'].shift(1) - data['close'].shift(2))
    data['volume_accel'] = (data['volume'] - data['volume'].shift(1)) - (data['volume'].shift(1) - data['volume'].shift(2))
    data['short_divergence'] = data['price_accel'] / (data['volume_accel'] + 1e-8)
    
    # Medium-Term Divergence (8-day)
    def linear_regression_slope(series, window):
        slopes = []
        for i in range(len(series)):
            if i >= window - 1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    data['price_momentum_slope'] = linear_regression_slope(data['close'], 8)
    data['volume_momentum_slope'] = linear_regression_slope(data['volume'], 8)
    data['medium_divergence'] = data['price_momentum_slope'] / (data['volume_momentum_slope'] + 1e-8)
    
    # Long-Term Divergence (15-day)
    data['price_vol_short'] = data['close'].rolling(window=15, min_periods=1).std()
    data['price_vol_long'] = data['close'].rolling(window=15, min_periods=1).std().shift(15)
    data['volume_vol_short'] = data['volume'].rolling(window=15, min_periods=1).std()
    data['volume_vol_long'] = data['volume'].rolling(window=15, min_periods=1).std().shift(15)
    
    data['price_vol_trend'] = data['price_vol_short'] - data['price_vol_long']
    data['volume_vol_trend'] = data['volume_vol_short'] - data['volume_vol_long']
    data['long_divergence'] = data['price_vol_trend'] / (data['volume_vol_trend'] + 1e-8)
    
    # Analyze Momentum Acceleration Patterns
    # Price Momentum Characteristics
    data['price_accel_sign'] = np.sign(data['price_accel'])
    data['accel_persistence'] = 0
    for i in range(2, len(data)):
        if data['price_accel_sign'].iloc[i] == data['price_accel_sign'].iloc[i-1]:
            data.loc[data.index[i], 'accel_persistence'] = data['accel_persistence'].iloc[i-1] + 1
    
    data['momentum_strength'] = data['price_accel'].rolling(window=5, min_periods=1).sum() / \
                               (data['price_accel'].abs().rolling(window=5, min_periods=1).sum() + 1e-8)
    data['momentum_quality'] = data['accel_persistence'] * data['momentum_strength']
    
    # Volume Momentum Dynamics
    data['volume_accel_trend'] = linear_regression_slope(data['volume_accel'], 7)
    data['volume_momentum_vol'] = data['volume_accel'].rolling(window=5, min_periods=1).std() / \
                                 (data['volume_accel'].abs().rolling(window=5, min_periods=1).mean() + 1e-8)
    data['volume_momentum_stability'] = data['volume_accel_trend'] / (data['volume_momentum_vol'] + 1e-8)
    
    # Assess Divergence-Momentum Interactions
    # Multi-Timeframe Divergence Patterns
    divergence_columns = ['short_divergence', 'medium_divergence', 'long_divergence']
    data['divergence_consistency'] = 0
    for col in divergence_columns:
        data['divergence_consistency'] += ((data[col] > 2.0) | (data[col] < 0.5)).astype(int)
    
    # Detect Momentum Reversal Signals
    data['divergence_extreme'] = ((data['short_divergence'] > 4.0) | (data['short_divergence'] < 0.25) |
                                 (data['medium_divergence'] > 4.0) | (data['medium_divergence'] < 0.25) |
                                 (data['long_divergence'] > 4.0) | (data['long_divergence'] < 0.25)).astype(int)
    
    # Construct Divergence-Acceleration Components
    # Apply filters and weightings
    data['short_div_factor'] = data['short_divergence'] * data['momentum_quality'] * \
                              (1 + data['volume_momentum_stability']) * (1 - data['divergence_extreme'] * 0.5)
    
    data['medium_div_factor'] = data['medium_divergence'] * data['momentum_quality'] * \
                               (1 + data['volume_momentum_stability']) * (1 - data['divergence_extreme'] * 0.5)
    
    data['long_div_factor'] = data['long_divergence'] * data['momentum_quality'] * \
                             (1 + data['volume_momentum_stability']) * (1 - data['divergence_extreme'] * 0.5)
    
    # Generate Momentum Reversal Forecasts
    data['max_accel_persistence'] = data['accel_persistence'].expanding().max()
    data['momentum_fatigue'] = data['accel_persistence'] / (data['max_accel_persistence'] + 1e-8)
    
    # Calculate reversal probability
    data['momentum_duration'] = data['accel_persistence'].rolling(window=10, min_periods=1).mean()
    data['divergence_alignment'] = (data['short_divergence'] * data['medium_divergence'] * data['long_divergence']).abs()
    data['reversal_probability'] = data['momentum_fatigue'] * (1 / (data['divergence_alignment'] + 1e-8))
    
    # Generate Final Alpha Factor
    # Combine all components
    weights = [0.4, 0.35, 0.25]  # Short, medium, long term weights
    data['combined_divergence'] = (weights[0] * data['short_div_factor'] + 
                                  weights[1] * data['medium_div_factor'] + 
                                  weights[2] * data['long_div_factor'])
    
    # Apply final adjustments
    data['alpha_factor'] = (data['combined_divergence'] * 
                           (1 - data['reversal_probability']) * 
                           (1 + data['divergence_consistency'] / 3) * 
                           (1 - data['momentum_fatigue']))
    
    # Clean and return
    alpha_series = data['alpha_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
