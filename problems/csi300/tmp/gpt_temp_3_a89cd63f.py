import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    High-Low Range Momentum Divergence factor that captures the divergence between
    short-term and medium-term volatility momentum, confirmed by volume trends.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily high-low range
    data['daily_range'] = data['high'] - data['low']
    
    # Short-term range momentum (t-4 to t)
    short_term_ranges = []
    for i in range(len(data)):
        if i >= 4:
            window_ranges = data['daily_range'].iloc[i-4:i+1].values
            if len(window_ranges) == 5:
                x = np.arange(5)
                slope, _, _, _, _ = stats.linregress(x, window_ranges)
                short_term_ranges.append(slope)
            else:
                short_term_ranges.append(np.nan)
        else:
            short_term_ranges.append(np.nan)
    
    data['short_term_slope'] = short_term_ranges
    
    # Medium-term range momentum (t-9 to t-5)
    medium_term_ranges = []
    for i in range(len(data)):
        if i >= 9:
            window_ranges = data['daily_range'].iloc[i-9:i-4].values
            if len(window_ranges) == 5:
                x = np.arange(5)
                slope, _, _, _, _ = stats.linregress(x, window_ranges)
                medium_term_ranges.append(slope)
            else:
                medium_term_ranges.append(np.nan)
        else:
            medium_term_ranges.append(np.nan)
    
    data['medium_term_slope'] = medium_term_ranges
    
    # Calculate divergence between short-term and medium-term momentum
    data['momentum_divergence'] = data['short_term_slope'] - data['medium_term_slope']
    
    # Volume trend alignment (t-4 to t)
    volume_trends = []
    for i in range(len(data)):
        if i >= 4:
            window_volume = data['volume'].iloc[i-4:i+1].values
            if len(window_volume) == 5:
                x = np.arange(5)
                volume_slope, _, _, _, _ = stats.linregress(x, window_volume)
                volume_trends.append(volume_slope)
            else:
                volume_trends.append(np.nan)
        else:
            volume_trends.append(np.nan)
    
    data['volume_trend'] = volume_trends
    
    # Calculate correlation between volume trend and range momentum
    volume_momentum_corr = []
    for i in range(len(data)):
        if i >= 8:  # Need enough data for correlation calculation
            volume_window = data['volume'].iloc[i-4:i+1].values
            range_window = data['daily_range'].iloc[i-4:i+1].values
            if len(volume_window) == 5 and len(range_window) == 5:
                corr = np.corrcoef(volume_window, range_window)[0, 1]
                volume_momentum_corr.append(corr if not np.isnan(corr) else 0)
            else:
                volume_momentum_corr.append(0)
        else:
            volume_momentum_corr.append(0)
    
    data['volume_momentum_corr'] = volume_momentum_corr
    
    # Absolute volume level (normalized)
    data['volume_rank'] = data['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, 
        raw=False
    )
    
    # Final factor calculation
    # Combine momentum divergence with volume confirmation
    data['factor'] = (
        data['momentum_divergence'] * 
        np.sign(data['volume_trend']) *  # Volume trend direction
        (1 + data['volume_momentum_corr']) *  # Volume-range correlation strength
        (1 + np.tanh(data['volume_rank'] / 10))  # Volume level adjustment
    )
    
    # Handle NaN values
    data['factor'] = data['factor'].fillna(0)
    
    return data['factor']
