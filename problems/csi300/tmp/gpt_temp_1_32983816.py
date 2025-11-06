import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate price acceleration (second derivative of Close prices)
    close_returns = df['close'].pct_change()
    price_acceleration = close_returns.diff().rolling(window=5, min_periods=3).mean()
    
    # Calculate VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(window=20, min_periods=10).sum() / df['volume'].rolling(window=20, min_periods=10).sum()
    
    # Calculate VWAP slope (20-day linear regression slope)
    def vwap_slope_calc(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    vwap_slope = vwap.rolling(window=20, min_periods=10).apply(vwap_slope_calc, raw=False)
    
    # Calculate volume z-score relative to 20-day moving average
    volume_ma = df['volume'].rolling(window=20, min_periods=10).mean()
    volume_std = df['volume'].rolling(window=20, min_periods=10).std()
    volume_zscore = (df['volume'] - volume_ma) / volume_std
    
    # Identify divergence patterns
    price_accel_ma = price_acceleration.rolling(window=5, min_periods=3).mean()
    vwap_slope_ma = vwap_slope.rolling(window=5, min_periods=3).mean()
    
    # Positive divergence: price acceleration decreasing, VWAP slope increasing
    positive_divergence = (price_acceleration < price_accel_ma) & (vwap_slope > vwap_slope_ma)
    
    # Negative divergence: price acceleration increasing, VWAP slope decreasing
    negative_divergence = (price_acceleration > price_accel_ma) & (vwap_slope < vwap_slope_ma)
    
    # Generate factor signals
    factor = pd.Series(0.0, index=df.index)
    
    # Strong buy: positive divergence with high volume (z-score > 0.5)
    factor[positive_divergence & (volume_zscore > 0.5)] = 2.0
    
    # Weak buy: positive divergence with low volume (z-score <= 0.5)
    factor[positive_divergence & (volume_zscore <= 0.5)] = 1.0
    
    # Strong sell: negative divergence with high volume (z-score > 0.5)
    factor[negative_divergence & (volume_zscore > 0.5)] = -2.0
    
    # Weak sell: negative divergence with low volume (z-score <= 0.5)
    factor[negative_divergence & (volume_zscore <= 0.5)] = -1.0
    
    return factor
