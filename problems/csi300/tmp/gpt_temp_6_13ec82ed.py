import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical indicators:
    - High-Low Volatility Persistence
    - Volume Breakout Momentum  
    - Intraday Reversal Strength
    - Price-Volume Divergence Factor
    - Amplitude-Adjusted Acceleration
    - Volume-Clustered Order Flow
    """
    # High-Low Volatility Persistence
    df['high_low_range'] = df['high'] - df['low']
    
    # Recent period volatility (t-5 to t-1)
    recent_vol = df['high_low_range'].rolling(window=5, min_periods=3).mean().shift(1)
    
    # Historical baseline volatility (t-20 to t-6)
    historical_vol = df['high_low_range'].rolling(window=15, min_periods=10).mean().shift(6)
    
    # Volatility persistence ratio with smoothing
    vol_persistence = (recent_vol / historical_vol).rolling(window=3, min_periods=2).mean()
    
    # Volume Breakout Momentum
    volume_mean = df['volume'].rolling(window=20, min_periods=15).mean().shift(1)
    volume_std = df['volume'].rolling(window=20, min_periods=15).std().shift(1)
    volume_zscore = (df['volume'] - volume_mean) / (volume_std + 1e-8)
    
    price_change = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    volume_momentum = price_change * volume_zscore
    
    # Intraday Reversal Strength
    intraday_return = (df['close'] - df['open']) / df['open']
    prev_day_return = (df['close'].shift(1) - df['close'].shift(2)) / df['close'].shift(2)
    
    reversal_strength = -np.sign(prev_day_return) * intraday_return * np.abs(prev_day_return)
    reversal_strength = reversal_strength * (df['volume'] / volume_mean)
    
    # Price-Volume Divergence Factor
    def linear_slope(series):
        x = np.arange(len(series))
        return stats.linregress(x, series.values)[0] if len(series) >= 3 else 0
    
    price_slope = df['close'].rolling(window=6, min_periods=4).apply(linear_slope, raw=False)
    volume_slope = df['volume'].rolling(window=6, min_periods=4).apply(linear_slope, raw=False)
    
    price_volume_divergence = np.sign(price_slope) * np.sign(volume_slope) * np.abs(price_slope - volume_slope)
    
    # Amplitude-Adjusted Acceleration
    returns = df['close'].pct_change()
    acceleration = returns.diff()
    
    recent_volatility = df['high_low_range'].rolling(window=10, min_periods=7).mean().shift(1)
    normalized_acceleration = acceleration / (recent_volatility + 1e-8)
    amplitude_adjusted_accel = normalized_acceleration.rolling(window=3, min_periods=2).mean()
    
    # Volume-Clustered Order Flow
    volume_percentile = df['volume'].rolling(window=20, min_periods=15).rank(pct=True)
    
    high_volume_mask = volume_percentile > 0.8
    high_volume_returns = df['close'].pct_change().where(high_volume_mask.shift(1))
    volume_cluster_effect = high_volume_returns.rolling(window=5, min_periods=3).mean()
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'vol_persistence': vol_persistence,
        'volume_momentum': volume_momentum,
        'reversal_strength': reversal_strength,
        'price_volume_divergence': price_volume_divergence,
        'amplitude_adjusted_accel': amplitude_adjusted_accel,
        'volume_cluster_effect': volume_cluster_effect
    })
    
    # Standardize each factor and combine
    combined_factor = factors.apply(lambda x: (x - x.rolling(window=50, min_periods=30).mean()) / 
                                   (x.rolling(window=50, min_periods=30).std() + 1e-8)).mean(axis=1)
    
    return combined_factor
