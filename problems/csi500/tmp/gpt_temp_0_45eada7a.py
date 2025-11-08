import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Price-Volume Fractal Divergence factor that compares multi-scale price and volume trends
    to identify predictive divergence patterns across different timeframes.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Function to calculate linear slope for a given window
    def calc_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                if len(y) == window and not np.isnan(y).any():
                    slope, _, _, _, _ = stats.linregress(x, y)
                    slopes.iloc[i] = slope
        return slopes
    
    # Calculate multi-scale price trends
    price_short = calc_slope(data['close'], 3)
    price_medium = calc_slope(data['close'], 10)
    price_long = calc_slope(data['close'], 30)
    
    # Calculate multi-scale volume trends
    volume_short = calc_slope(data['volume'], 3)
    volume_medium = calc_slope(data['volume'], 10)
    volume_long = calc_slope(data['volume'], 30)
    
    # Calculate divergence scores for each timeframe
    def calc_divergence(price_slope, volume_slope):
        # Normalize slopes by their recent volatility
        price_vol = price_slope.rolling(window=20, min_periods=1).std()
        volume_vol = volume_slope.rolling(window=20, min_periods=1).std()
        
        # Avoid division by zero
        price_vol = price_vol.replace(0, np.nan)
        volume_vol = volume_vol.replace(0, np.nan)
        
        normalized_price = price_slope / price_vol
        normalized_volume = volume_slope / volume_vol
        
        # Calculate divergence as product of normalized slopes
        # Negative product indicates divergence (opposite directions)
        divergence = -normalized_price * normalized_volume
        return divergence
    
    # Calculate divergences for each timeframe
    div_short = calc_divergence(price_short, volume_short)
    div_medium = calc_divergence(price_medium, volume_medium)
    div_long = calc_divergence(price_long, volume_long)
    
    # Calculate trend alignment hierarchy
    def calc_trend_alignment(price_slope, volume_slope):
        # Convert slopes to binary directions
        price_dir = np.sign(price_slope)
        volume_dir = np.sign(volume_slope)
        
        # Calculate alignment (1 for same direction, -1 for opposite)
        alignment = price_dir * volume_dir
        return alignment
    
    # Calculate alignment for each timeframe
    align_short = calc_trend_alignment(price_short, volume_short)
    align_medium = calc_trend_alignment(price_medium, volume_medium)
    align_long = calc_trend_alignment(price_long, volume_long)
    
    # Calculate hierarchical convergence score
    # Higher score when longer-term trends dominate and align
    hierarchy_score = (
        align_long.fillna(0) * 0.5 + 
        align_medium.fillna(0) * 0.3 + 
        align_short.fillna(0) * 0.2
    )
    
    # Calculate composite divergence score with timeframe weights
    # Short-term: 0.2, Medium-term: 0.3, Long-term: 0.5
    composite_divergence = (
        div_short.fillna(0) * 0.2 + 
        div_medium.fillna(0) * 0.3 + 
        div_long.fillna(0) * 0.5
    )
    
    # Incorporate price-level context
    # Use normalized price position within recent range
    price_position = (data['close'] - data['close'].rolling(window=20).min()) / \
                    (data['close'].rolling(window=20).max() - data['close'].rolling(window=20).min())
    price_position = price_position.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    
    # Adjust for recent volatility regime
    recent_volatility = data['close'].pct_change().rolling(window=20).std()
    vol_normalized = recent_volatility / recent_volatility.rolling(window=60).mean()
    vol_normalized = vol_normalized.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # Final factor calculation
    # Strong positive values indicate bullish divergence (price up, volume down)
    # Strong negative values indicate bearish divergence (price down, volume up)
    factor = composite_divergence * hierarchy_score
    
    # Adjust for price position (divergence more significant at extremes)
    price_extremity = np.abs(price_position - 0.5) * 2  # 0 at middle, 1 at extremes
    factor = factor * (1 + price_extremity)
    
    # Adjust for volatility regime (divergence more significant in low vol)
    vol_adjustment = 1.0 / np.sqrt(vol_normalized.clip(lower=0.1))
    factor = factor * vol_adjustment
    
    # Normalize the final factor
    factor_normalized = (factor - factor.rolling(window=60).mean()) / factor.rolling(window=60).std()
    factor_normalized = factor_normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factor_normalized
