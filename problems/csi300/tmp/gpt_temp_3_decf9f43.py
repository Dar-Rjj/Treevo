import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Price-Volume Divergence with Regime Detection alpha factor
    """
    data = df.copy()
    
    # Avoid division by zero
    epsilon = 1e-8
    
    # 1. Calculate Price-Volume Divergence
    # Price-based Momentum
    # Intraday Momentum
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'] + epsilon)
    
    # Multi-timeframe Price Strength
    # 5-day price range efficiency
    data['high_5d'] = data['high'].rolling(window=5, min_periods=3).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=3).min()
    data['price_range_efficiency'] = (data['close'] - data['low_5d']) / (data['high_5d'] - data['low_5d'] + epsilon)
    
    # 20-day price persistence
    data['close_gt_open'] = (data['close'] > data['open']).astype(int)
    data['price_persistence'] = data['close_gt_open'].rolling(window=20, min_periods=10).mean()
    
    # Combine price momentum components
    data['price_momentum'] = (
        0.4 * data['intraday_momentum'] + 
        0.4 * data['price_range_efficiency'] + 
        0.2 * data['price_persistence']
    )
    
    # Volume-based Momentum
    # Volume Spike Detection
    data['volume_median_20d'] = data['volume'].rolling(window=20, min_periods=10).median()
    data['volume_spike'] = data['volume'] / (data['volume_median_20d'] + epsilon)
    
    # Volume Trend Consistency
    # 5-day volume slope
    def volume_slope(series):
        if len(series) < 3:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    data['volume_slope_5d'] = data['volume'].rolling(window=5, min_periods=3).apply(
        volume_slope, raw=True
    )
    
    # Volume direction persistence
    data['volume_change'] = data['volume'].pct_change()
    data['volume_increase'] = (data['volume_change'] > 0).astype(int)
    
    def consecutive_volume_direction(series):
        if len(series) < 2:
            return 0
        current_dir = series.iloc[-1]
        consecutive = 0
        for i in range(len(series)-2, -1, -1):
            if series.iloc[i] == current_dir:
                consecutive += 1
            else:
                break
        return consecutive * (1 if current_dir == 1 else -1)
    
    data['volume_direction_persistence'] = data['volume_increase'].rolling(
        window=10, min_periods=5
    ).apply(consecutive_volume_direction, raw=False)
    
    # Combine volume momentum components
    data['volume_momentum'] = (
        0.5 * data['volume_spike'] + 
        0.3 * data['volume_slope_5d'] + 
        0.2 * data['volume_direction_persistence']
    )
    
    # 2. Detect Market Regime Context
    # Volatility Regime
    data['daily_return'] = data['close'].pct_change()
    data['realized_volatility_20d'] = data['daily_return'].rolling(
        window=20, min_periods=10
    ).std()
    
    # Volatility regime classification
    vol_median = data['realized_volatility_20d'].rolling(window=60, min_periods=30).median()
    data['high_vol_regime'] = (data['realized_volatility_20d'] > vol_median).astype(int)
    
    # Trend Regime
    def trend_strength(series):
        if len(series) < 10:
            return np.nan, np.nan
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        return slope, r_value**2
    
    trend_results = data['close'].rolling(window=20, min_periods=10).apply(
        lambda x: trend_strength(x)[0] if len(x) >= 10 else np.nan, raw=True
    )
    r_squared_results = data['close'].rolling(window=20, min_periods=10).apply(
        lambda x: trend_strength(x)[1] if len(x) >= 10 else np.nan, raw=True
    )
    
    data['price_slope_20d'] = trend_results
    data['trend_r_squared'] = r_squared_results
    
    # Trend regime classification
    data['trending_market'] = (
        (data['trend_r_squared'] > 0.3) & 
        (np.abs(data['price_slope_20d']) > 0.001)
    ).astype(int)
    data['uptrend'] = (data['price_slope_20d'] > 0).astype(int)
    
    # 3. Generate Adaptive Alpha Signal
    # Calculate Divergence Score
    # Normalize price and volume momentum
    data['price_momentum_z'] = (
        data['price_momentum'] - data['price_momentum'].rolling(window=60, min_periods=30).mean()
    ) / (data['price_momentum'].rolling(window=60, min_periods=30).std() + epsilon)
    
    data['volume_momentum_z'] = (
        data['volume_momentum'] - data['volume_momentum'].rolling(window=60, min_periods=30).mean()
    ) / (data['volume_momentum'].rolling(window=60, min_periods=30).std() + epsilon)
    
    # Divergence detection
    data['momentum_aligned'] = (
        (data['price_momentum_z'] > 0) & (data['volume_momentum_z'] > 0) |
        (data['price_momentum_z'] < 0) & (data['volume_momentum_z'] < 0)
    ).astype(int)
    
    data['divergence_magnitude'] = np.abs(data['price_momentum_z'] - data['volume_momentum_z'])
    
    # Calculate correlation persistence
    data['momentum_correlation_20d'] = data['price_momentum_z'].rolling(
        window=20, min_periods=10
    ).corr(data['volume_momentum_z'])
    
    # Apply Regime-Based Filtering
    # Base divergence score
    data['base_divergence'] = data['divergence_magnitude'] * (1 - data['momentum_aligned'])
    
    # High volatility regime adjustments
    high_vol_weight = np.where(
        data['high_vol_regime'] == 1,
        data['volume_spike'] * 1.5,  # Require stronger volume confirmation
        1.0
    )
    
    # Low volatility regime adjustments
    low_vol_weight = np.where(
        data['high_vol_regime'] == 0,
        data['volume_spike'] * 0.7,  # Allow weaker volume signals
        1.0
    )
    
    # Trending market adjustments
    trend_weight = np.where(
        data['trending_market'] == 1,
        np.abs(data['price_slope_20d']) * 2.0,  # Amplify signals in trending markets
        1.0
    )
    
    # Final alpha signal
    data['alpha_signal'] = (
        data['base_divergence'] * 
        high_vol_weight * 
        low_vol_weight * 
        trend_weight * 
        (1 - np.abs(data['momentum_correlation_20d']))  # Higher weight when correlation is low
    )
    
    # Direction based on price momentum
    data['alpha_signal'] = data['alpha_signal'] * np.sign(data['price_momentum_z'])
    
    # Clean up intermediate columns
    result = data['alpha_signal'].copy()
    
    return result
