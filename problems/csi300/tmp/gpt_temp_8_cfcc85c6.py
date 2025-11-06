import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily price range
    daily_range = df['high'] - df['low']
    
    # Calculate daily returns from close prices
    daily_returns = df['close'].pct_change()
    
    # Calculate rolling volatility measures (21-day window)
    range_volatility = daily_range.rolling(window=21).std()
    returns_volatility = daily_returns.rolling(window=21).std()
    
    # Combine both volatility measures using geometric mean
    combined_volatility = np.sqrt(range_volatility * returns_volatility)
    
    # Calculate Z-score of current volatility relative to historical (63-day lookback)
    volatility_zscore = (combined_volatility - combined_volatility.rolling(window=63).mean()) / combined_volatility.rolling(window=63).std()
    
    # Calculate volatility slope using linear regression over 5 days
    def calc_slope(series):
        if len(series) < 5 or series.isna().any():
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]
    
    volatility_slope = combined_volatility.rolling(window=5).apply(calc_slope, raw=False)
    
    # Detect regime changes
    high_vol_regime = volatility_zscore > 1.0
    low_vol_regime = volatility_zscore < -1.0
    increasing_vol = volatility_slope > 0
    decreasing_vol = volatility_slope < 0
    
    # Generate trading signals
    # High to low volatility transition (long signal for breakout potential)
    high_to_low = (high_vol_regime.shift(1)) & (decreasing_vol) & (volatility_zscore < 0.5)
    
    # Low to high volatility transition (short signal for mean reversion)
    low_to_high = (low_vol_regime.shift(1)) & (increasing_vol) & (volatility_zscore > -0.5)
    
    # Combine signals with weights
    factor = pd.Series(0.0, index=df.index)
    factor[high_to_low] = 1.0
    factor[low_to_high] = -1.0
    
    # Add momentum to the signal based on volatility slope strength
    factor = factor + (volatility_slope * 0.5)
    
    return factor
