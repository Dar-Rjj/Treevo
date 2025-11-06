import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price returns
    data['prev_close'] = data['close'].shift(1)
    data['daily_return'] = (data['close'] - data['prev_close']) / data['prev_close']
    
    # 1. Calculate Asymmetric Price Movement Components
    # Upside Price Momentum
    data['max_daily_return'] = (data['high'] - data['prev_close']) / data['prev_close']
    
    # Upside acceleration calculations
    data['upside_return_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['upside_return_10d'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    data['upside_acceleration'] = data['upside_return_5d'] - (data['upside_return_10d'] - data['upside_return_5d'])
    
    # Downside Price Momentum
    data['min_daily_return'] = (data['low'] - data['prev_close']) / data['prev_close']
    
    # Downside acceleration calculations
    data['downside_return_5d'] = (data['close'].shift(5) - data['close']) / data['close']
    data['downside_return_10d'] = (data['close'].shift(10) - data['close']) / data['close']
    data['downside_acceleration'] = data['downside_return_5d'] - (data['downside_return_10d'] - data['downside_return_5d'])
    
    # 2. Calculate Volume Asymmetry Components
    # Identify up and down days
    data['is_up_day'] = data['close'] > data['open']
    data['is_down_day'] = data['close'] < data['open']
    
    # Upside volume concentration
    data['upside_volume'] = data['volume'] * data['is_up_day']
    data['upside_volume_5d'] = data['upside_volume'].rolling(window=5, min_periods=1).sum()
    
    # Upside volume momentum
    data['upside_volume_roc_5d'] = (data['upside_volume_5d'] - data['upside_volume_5d'].shift(5)) / data['upside_volume_5d'].shift(5)
    data['upside_volume_roc_10d'] = (data['upside_volume_5d'] - data['upside_volume_5d'].shift(10)) / data['upside_volume_5d'].shift(10)
    
    # Downside volume concentration
    data['downside_volume'] = data['volume'] * data['is_down_day']
    data['downside_volume_5d'] = data['downside_volume'].rolling(window=5, min_periods=1).sum()
    
    # Downside volume momentum
    data['downside_volume_roc_5d'] = (data['downside_volume_5d'] - data['downside_volume_5d'].shift(5)) / data['downside_volume_5d'].shift(5)
    data['downside_volume_roc_10d'] = (data['downside_volume_5d'] - data['downside_volume_5d'].shift(10)) / data['downside_volume_5d'].shift(10)
    
    # Volume asymmetry ratio
    data['volume_asymmetry_ratio'] = np.log(data['upside_volume_5d'] / (data['downside_volume_5d'] + 1e-8))
    
    # 3. Detect Market Regime Patterns
    # High volatility regime
    data['daily_range'] = (data['high'] - data['low']) / data['prev_close']
    data['avg_range_20d'] = data['daily_range'].rolling(window=20, min_periods=10).mean()
    data['range_ratio'] = data['daily_range'] / (data['avg_range_20d'] + 1e-8)
    data['high_vol_regime'] = data['range_ratio'] > 1.5
    
    # Trend regime
    def calc_slope(window):
        if len(window) < 5:
            return 0
        x = np.arange(len(window))
        slope, _, _, _, _ = stats.linregress(x, window)
        return slope
    
    data['price_slope_20d'] = data['close'].rolling(window=20, min_periods=10).apply(calc_slope, raw=True)
    data['trend_regime'] = np.where(data['price_slope_20d'] > 0.001, 'uptrend', 
                                   np.where(data['price_slope_20d'] < -0.001, 'downtrend', 'sideways'))
    
    # Volume regime
    data['volume_median_10d'] = data['volume'].rolling(window=10, min_periods=5).median()
    data['volume_spike_ratio'] = data['volume'] / (data['volume_median_10d'] + 1e-8)
    data['high_volume_regime'] = data['volume_spike_ratio'] > 1.2
    
    # 4. Generate Regime-Adaptive Alpha Signal
    # Asymmetric momentum score
    data['upside_momentum'] = data['max_daily_return'] * data['upside_acceleration']
    data['downside_momentum'] = data['min_daily_return'] * data['downside_acceleration']
    data['asymmetry_difference'] = (data['upside_momentum'] * data['upside_volume_5d'] - 
                                   data['downside_momentum'] * data['downside_volume_5d'])
    
    # Regime-based weighting
    # High volatility regime weight
    data['volatility_weight'] = data['range_ratio'] * data['volume_asymmetry_ratio']
    
    # Trend regime weight
    data['trend_weight'] = np.where(data['trend_regime'] == 'uptrend', 1.2,
                                   np.where(data['trend_regime'] == 'downtrend', 0.8, 1.0))
    
    # Volume regime weight
    data['volume_weight'] = np.where(data['high_volume_regime'], 1.3, 0.7)
    
    # Combined regime weight
    data['regime_weight'] = data['volatility_weight'] * data['trend_weight'] * data['volume_weight']
    
    # Volatility adjustment
    # Calculate conditional volatility
    data['up_day_returns'] = data['daily_return'] * data['is_up_day']
    data['down_day_returns'] = data['daily_return'] * data['is_down_day']
    
    data['upside_volatility'] = data['up_day_returns'].rolling(window=10, min_periods=5).std()
    data['downside_volatility'] = data['down_day_returns'].rolling(window=10, min_periods=5).std()
    
    # Scale signal by volatility asymmetry
    data['volatility_scaling'] = np.where(
        data['asymmetry_difference'] > 0,
        data['asymmetry_difference'] / (data['upside_volatility'] + 1e-8),
        data['asymmetry_difference'] / (data['downside_volatility'] + 1e-8)
    )
    
    # Final alpha signal
    data['alpha_signal'] = data['volatility_scaling'] * data['regime_weight']
    
    # Clean up and return
    alpha_series = data['alpha_signal'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_series
