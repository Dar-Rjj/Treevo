import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Convergence factor combining short-term momentum
    and medium-term trend signals with convergence analysis.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Short-Term Momentum (3-day)
    # Price Momentum
    close_roc_3d = data['close'].pct_change(2)  # t-2 to t
    high_low_range = data['high'] - data['low']
    range_change_3d = high_low_range.pct_change(2)
    
    # Volume Momentum
    volume_roc_3d = data['volume'].pct_change(2)
    volume_var_3d = data['volume'].rolling(window=3).var()
    
    # Medium-Term Trend (10-day)
    # Price Trend Strength
    def calculate_slope(series, window):
        x = np.arange(window)
        slopes = series.rolling(window=window).apply(
            lambda y: np.polyfit(x, y, 1)[0] if not np.isnan(y).any() else np.nan,
            raw=False
        )
        return slopes
    
    close_slope_10d = calculate_slope(data['close'], 10)
    range_stability_10d = high_low_range.rolling(window=10).std()
    
    # Volume Trend Pattern
    volume_slope_10d = calculate_slope(data['volume'], 10)
    
    # Volume regime shifts (detect significant changes in volume pattern)
    volume_ma_10d = data['volume'].rolling(window=10).mean()
    volume_std_10d = data['volume'].rolling(window=10).std()
    volume_regime_shift = (data['volume'] - volume_ma_10d).abs() > (2 * volume_std_10d)
    
    # Convergence Analysis
    # Price momentum alignment (short vs medium term)
    price_alignment = np.sign(close_roc_3d) * np.sign(close_slope_10d)
    
    # Volume momentum alignment
    volume_alignment = np.sign(volume_roc_3d) * np.sign(volume_slope_10d)
    
    # Generate Convergence Factor
    # Base signals
    price_signal = close_roc_3d * np.abs(close_slope_10d)
    volume_signal = volume_roc_3d * np.abs(volume_slope_10d)
    
    # Alignment weights
    price_weight = np.where(price_alignment > 0, 1.5, 0.5)
    volume_weight = np.where(volume_alignment > 0, 1.5, 0.5)
    
    # Combined convergence factor
    convergence_factor = (
        price_signal * price_weight + 
        volume_signal * volume_weight +
        range_change_3d * 0.3 +
        volume_var_3d * 0.2
    )
    
    # Normalize the factor
    convergence_factor = (convergence_factor - convergence_factor.rolling(window=20).mean()) / convergence_factor.rolling(window=20).std()
    
    return convergence_factor
