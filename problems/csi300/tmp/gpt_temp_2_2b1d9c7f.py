import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple market microstructure signals
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Price-Volume Divergence Factor
    price_momentum = df['close'].pct_change(5)
    volume_trend = df['volume'].rolling(window=5).apply(
        lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == 5 else np.nan, 
        raw=False
    )
    pv_divergence = price_momentum * (1 / (volume_trend + 1e-8))
    
    # Range Efficiency Factor
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    avg_true_range = true_range.rolling(window=5).mean()
    avg_abs_return = abs(df['close'].pct_change()).rolling(window=5).mean()
    range_efficiency = avg_abs_return / (avg_true_range + 1e-8)
    
    # Volume-Scaled Reversal Factor
    returns_5d = df['close'].pct_change(5)
    returns_20d = df['close'].pct_change(20)
    
    # Calculate extreme moves (top/bottom 10%)
    extreme_threshold_high = returns_20d.rolling(window=20, min_periods=10).quantile(0.9)
    extreme_threshold_low = returns_20d.rolling(window=20, min_periods=10).quantile(0.1)
    
    is_extreme_high = returns_20d > extreme_threshold_high
    is_extreme_low = returns_20d < extreme_threshold_low
    is_extreme = is_extreme_high | is_extreme_low
    
    # Volume rank (percentile over 20 days)
    volume_rank = df['volume'].rolling(window=20).rank(pct=True)
    
    volume_reversal = -returns_5d * volume_rank
    volume_reversal = volume_reversal.where(is_extreme, 0)
    
    # Order Flow Persistence Factor
    direction = np.sign(df['close'] - df['close'].shift(1))
    directional_flow = df['amount'] * direction
    
    # Count consecutive same-sign flow days
    consecutive_days = directional_flow.groupby(
        (direction != direction.shift(1)).cumsum()
    ).cumcount() + 1
    
    cumulative_flow = directional_flow.rolling(window=5, min_periods=1).sum()
    order_persistence = consecutive_days * cumulative_flow
    
    # Volatility-Regime Volume Factor
    vol_10d = df['close'].pct_change().rolling(window=10).std()
    vol_median_20d = vol_10d.rolling(window=20).median()
    
    # Volatility regime
    is_high_vol = vol_10d > vol_median_20d
    is_low_vol = vol_10d <= vol_median_20d
    
    # Volume spikes (volume > 1.5 * 20-day average)
    vol_avg_20d = df['volume'].rolling(window=20).mean()
    volume_spikes = (df['volume'] > 1.5 * vol_avg_20d).rolling(window=5).sum()
    
    regime_multiplier = pd.Series(0, index=df.index)
    regime_multiplier[is_high_vol] = -1
    regime_multiplier[is_low_vol] = 1
    
    vol_regime_factor = volume_spikes * regime_multiplier
    
    # Combine factors with equal weights
    factors = pd.DataFrame({
        'pv_divergence': pv_divergence,
        'range_efficiency': range_efficiency,
        'volume_reversal': volume_reversal,
        'order_persistence': order_persistence,
        'vol_regime': vol_regime_factor
    })
    
    # Z-score normalize each factor and combine
    normalized_factors = factors.apply(lambda x: (x - x.rolling(window=20, min_periods=10).mean()) / 
                                     (x.rolling(window=20, min_periods=10).std() + 1e-8))
    
    result = normalized_factors.mean(axis=1)
    
    return result
