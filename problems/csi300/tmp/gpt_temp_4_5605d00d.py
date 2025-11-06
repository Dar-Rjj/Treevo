import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate Price Momentum
    # Compute Rolling Returns (5-day)
    price_momentum = df['close'].pct_change(periods=5)
    
    # Compute Rolling Volatility (10-day)
    price_volatility = df['close'].rolling(window=10).std()
    
    # Calculate Volume Momentum
    # Compute Volume Change (5-day)
    volume_change = df['volume'].pct_change(periods=5)
    
    # Compute Volume Trend (10-day linear regression slope)
    def calc_volume_slope(volume_series):
        if len(volume_series) < 10 or volume_series.isna().any():
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = linregress(x, volume_series)
        return slope
    
    volume_trend = df['volume'].rolling(window=10).apply(
        calc_volume_slope, raw=False
    )
    
    # Combine Signals
    # Detect Momentum Divergence
    price_momentum_direction = np.sign(price_momentum)
    volume_change_direction = np.sign(volume_change)
    
    # Generate divergence signal (1 when directions match, -1 when they diverge)
    divergence_signal = np.where(
        price_momentum_direction == volume_change_direction, 1, -1
    )
    
    # Generate Alpha Factor
    # Multiply price momentum by volume trend and adjust by divergence confirmation
    alpha_factor = (price_momentum * volume_trend * divergence_signal)
    
    # Normalize by price volatility to account for different volatility regimes
    normalized_alpha = alpha_factor / (price_volatility + 1e-8)
    
    return pd.Series(normalized_alpha, index=df.index)
