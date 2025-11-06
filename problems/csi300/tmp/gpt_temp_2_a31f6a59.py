import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Fractal Efficiency
    # Price Path Length: Sum of daily High-Low ranges over 10 days
    price_path_length = (data['high'] - data['low']).rolling(window=10, min_periods=10).sum()
    
    # Net Price Change: Absolute difference between Close(t) and Close(t-10)
    net_price_change = (data['close'] - data['close'].shift(10)).abs()
    
    # Efficiency Ratio: Net Price Change / Price Path Length
    efficiency_ratio = net_price_change / price_path_length
    
    # Incorporate Volume Confirmation
    # Volume Trend: 10-day volume slope via linear regression
    def calculate_volume_slope(volume_series):
        if len(volume_series) < 10:
            return np.nan
        X = np.arange(len(volume_series)).reshape(-1, 1)
        y = volume_series.values
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0]
    
    volume_slope = data['volume'].rolling(window=10, min_periods=10).apply(
        calculate_volume_slope, raw=False
    )
    
    # Volume-Weighted Efficiency: Efficiency Ratio × Volume Trend
    volume_weighted_efficiency = efficiency_ratio * volume_slope
    
    # Implement Regime Detection
    # Market Regime: Volatility clustering using 20-day High-Low range rolling window
    daily_range = data['high'] - data['low']
    volatility_regime = daily_range.rolling(window=20, min_periods=20).std()
    volatility_threshold = volatility_regime.rolling(window=60, min_periods=60).median()
    
    # Regime-Specific Adjustments
    high_vol_regime = volatility_regime > volatility_threshold
    low_vol_regime = volatility_regime <= volatility_threshold
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # High volatility: Emphasize volume confirmation (weight = 0.7)
    factor[high_vol_regime] = 0.7 * volume_weighted_efficiency[high_vol_regime] + 0.3 * efficiency_ratio[high_vol_regime]
    
    # Low volatility: Focus on efficiency signals (weight = 0.3)
    factor[low_vol_regime] = 0.3 * volume_weighted_efficiency[low_vol_regime] + 0.7 * efficiency_ratio[low_vol_regime]
    
    # Handle NaN values by forward filling
    factor = factor.fillna(method='ffill')
    
    return factor
