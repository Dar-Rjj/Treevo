import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate regime-adaptive alpha factor combining volatility-normalized momentum
    and volume divergence components.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Normalized Multi-Timeframe Momentum
    # Calculate short-term momentum (3-day)
    short_momentum = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    
    # Calculate medium-term momentum (10-day)
    medium_momentum = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    # Calculate 20-day rolling volatility
    daily_returns = data['close'].pct_change()
    volatility_20d = daily_returns.rolling(window=20).std()
    
    # Normalize momentums by volatility
    short_momentum_norm = short_momentum / volatility_20d
    medium_momentum_norm = medium_momentum / volatility_20d
    
    # Weighted combination (equal weights)
    momentum_factor = (short_momentum_norm + medium_momentum_norm) / 2
    
    # Volume Divergence Trend Factor
    def calculate_trend(series, window):
        """Calculate linear regression slope for given series and window."""
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y_values = series.iloc[i-window+1:i+1].values
                x_values = np.arange(window)
                if len(y_values) == window and not np.any(np.isnan(y_values)):
                    slope, _, _, _, _ = stats.linregress(x_values, y_values)
                    slopes.iloc[i] = slope
        return slopes
    
    # Price trend components
    price_trend_5d = calculate_trend(data['close'], 5)
    price_trend_10d = calculate_trend(data['close'], 10)
    
    # Volume trend components
    volume_trend_5d = calculate_trend(data['volume'], 5)
    volume_trend_10d = calculate_trend(data['volume'], 10)
    
    # Combine price and volume trends (average of 5d and 10d)
    price_trend_combined = (price_trend_5d + price_trend_10d) / 2
    volume_trend_combined = (volume_trend_5d + volume_trend_10d) / 2
    
    # Divergence detection with magnitude weighting
    volume_divergence = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if (not pd.isna(price_trend_combined.iloc[i]) and 
            not pd.isna(volume_trend_combined.iloc[i])):
            
            price_trend = price_trend_combined.iloc[i]
            volume_trend = volume_trend_combined.iloc[i]
            
            # Detect divergence and assign sign
            if price_trend > 0 and volume_trend < 0:  # Bearish divergence
                sign = -1
            elif price_trend < 0 and volume_trend > 0:  # Bullish divergence
                sign = 1
            else:  # No divergence or same direction
                sign = 0
            
            # Magnitude weighting (product of absolute trend magnitudes)
            magnitude = abs(price_trend) * abs(volume_trend)
            volume_divergence.iloc[i] = sign * magnitude
    
    # Regime-Adaptive Factor Blending
    # Calculate volatility regime
    volatility_10d = daily_returns.rolling(window=10).std()
    volatility_50d_avg = daily_returns.rolling(window=50).std().rolling(window=10).mean()
    
    high_vol_regime = volatility_10d > (1.2 * volatility_50d_avg)
    
    # Normalize factors before combination
    momentum_factor_norm = (momentum_factor - momentum_factor.rolling(50).mean()) / momentum_factor.rolling(50).std()
    volume_divergence_norm = (volume_divergence - volume_divergence.rolling(50).mean()) / volume_divergence.rolling(50).std()
    
    # Regime-dependent weights
    final_factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if (not pd.isna(momentum_factor_norm.iloc[i]) and 
            not pd.isna(volume_divergence_norm.iloc[i]) and
            not pd.isna(high_vol_regime.iloc[i])):
            
            if high_vol_regime.iloc[i]:
                # High volatility regime: favor volume divergence
                weight_momentum = 0.4
                weight_volume = 0.6
            else:
                # Normal volatility regime: balanced weights
                weight_momentum = 0.5
                weight_volume = 0.5
            
            final_factor.iloc[i] = (weight_momentum * momentum_factor_norm.iloc[i] + 
                                   weight_volume * volume_divergence_norm.iloc[i])
    
    return final_factor
