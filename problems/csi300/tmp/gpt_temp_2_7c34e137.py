import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple market microstructure signals.
    
    Parameters:
    df: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
        Index should be datetime
    
    Returns:
    pandas Series with factor values indexed by date
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Component 1: Intraday Trend Persistence Factor
    morning_strength = (df['high'] - df['open']) / df['open']
    afternoon_strength = (df['close'] - df['low']) / df['low']
    trend_persistence = morning_strength * afternoon_strength
    
    # Component 2: Volume-Weighted Price Acceleration
    price_accel = (df['close'] - df['close'].shift(1)) - (df['close'].shift(1) - df['close'].shift(2))
    volume_accel = (df['volume'] - df['volume'].shift(1)) - (df['volume'].shift(1) - df['volume'].shift(2))
    volume_weighted_accel = price_accel * volume_accel
    
    # Component 3: Overnight Gap Mean Reversion
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate 5-day average true range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    recent_volatility = true_range.rolling(window=5).mean()
    
    gap_reversion = -overnight_gap / (recent_volatility + 1e-8)  # Inverse for mean reversion
    
    # Component 4: Intraday Range Efficiency Factor
    total_movement = abs(df['high'] - df['low'])
    net_movement = abs(df['close'] - df['open'])
    range_efficiency = net_movement / (total_movement + 1e-8)
    
    # Component 5: Volume-Price Divergence Oscillator
    def calc_slope(series, window=5):
        """Calculate linear regression slope over rolling window"""
        x = np.arange(window)
        slopes = series.rolling(window=window).apply(
            lambda y: np.polyfit(x, y, 1)[0] if not y.isna().any() else np.nan,
            raw=False
        )
        return slopes
    
    price_slope = calc_slope(df['close'], 5)
    volume_slope = calc_slope(df['volume'], 5)
    divergence_oscillator = -price_slope * volume_slope  # Negative for divergence detection
    
    # Combine all components with equal weights
    components = pd.DataFrame({
        'trend_persistence': trend_persistence,
        'volume_weighted_accel': volume_weighted_accel,
        'gap_reversion': gap_reversion,
        'range_efficiency': range_efficiency,
        'divergence_oscillator': divergence_oscillator
    })
    
    # Z-score normalization for each component
    normalized_components = components.apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    
    # Final factor as weighted sum (equal weights)
    result = normalized_components.mean(axis=1)
    
    return result
