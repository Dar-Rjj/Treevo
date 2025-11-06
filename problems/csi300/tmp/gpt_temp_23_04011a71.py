import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum Divergence with Volume Confirmation
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Assessment
    # Calculate daily High-Low ranges
    data['daily_range'] = data['high'] - data['low']
    
    # Short-term volatility (5-day)
    data['vol_5d'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    
    # Medium-term volatility (20-day)
    data['vol_20d'] = data['daily_range'].rolling(window=20, min_periods=10).mean()
    
    # Volatility ratio to identify regime shifts
    data['vol_ratio'] = data['vol_5d'] / data['vol_20d']
    
    # Multi-Timeframe Momentum Analysis
    # Short-term momentum (3-day slope)
    def calc_slope_3d(series):
        if len(series) < 3:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    data['momentum_3d'] = data['close'].rolling(window=3, min_periods=3).apply(
        calc_slope_3d, raw=False
    )
    
    # Medium-term momentum (10-day slope)
    def calc_slope_10d(series):
        if len(series) < 10:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    data['momentum_10d'] = data['close'].rolling(window=10, min_periods=7).apply(
        calc_slope_10d, raw=False
    )
    
    # Momentum divergence score
    data['momentum_divergence'] = data['momentum_3d'] * data['momentum_10d']
    
    # Volume-Price Confirmation
    # Volume trend analysis (5-day slope)
    def calc_volume_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    data['volume_slope'] = data['volume'].rolling(window=5, min_periods=3).apply(
        calc_volume_slope, raw=False
    )
    
    # Intraday price behavior - Close position in High-Low range
    data['intraday_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['intraday_position'] = data['intraday_position'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-price alignment
    data['volume_price_alignment'] = data['volume_slope'] * data['intraday_position']
    
    # Adaptive Signal Generation
    # Combine momentum and volume components
    data['combined_signal'] = data['momentum_divergence'] * data['volume_price_alignment']
    
    # Apply volatility adjustment
    data['alpha_factor'] = data['combined_signal'] * data['vol_ratio']
    
    # Return the final alpha factor series
    return data['alpha_factor']
