import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Momentum-Adjusted Volume Acceleration
    # Compute Momentum Component
    momentum_5d = df['close'] / df['close'].shift(5) - 1
    momentum_10d = df['close'] / df['close'].shift(10) - 1
    
    # Compute Volume Acceleration
    volume_ma_3d = df['volume'].rolling(window=3).mean()
    volume_ma_10d = df['volume'].rolling(window=10).mean()
    volume_acceleration = volume_ma_3d / volume_ma_10d - 1
    
    # Combine Momentum and Volume Components
    momentum_volume_5d = momentum_5d * volume_acceleration
    momentum_volume_10d = momentum_10d * volume_acceleration
    factor1 = (momentum_volume_5d + momentum_volume_10d) / 2
    
    # High-Low Range Breakout with Volume Confirmation
    # Calculate Price Range Component
    high_20d = df['high'].rolling(window=20).max()
    low_20d = df['low'].rolling(window=20).min()
    range_20d = high_20d - low_20d
    current_range = df['high'] - df['low']
    range_ratio = current_range / range_20d
    
    # Calculate Volume Confirmation
    volume_5d = df['volume'].rolling(window=5)
    volume_rank = volume_5d.rank(ascending=False, pct=True)
    
    # Combine Range and Volume Signals
    range_volume = range_ratio * volume_rank
    momentum_sign = np.sign(momentum_5d)
    factor2 = range_volume * momentum_sign
    
    # Volatility-Regime Adjusted Return
    # Identify Volatility Regime
    returns = df['close'].pct_change()
    volatility_20d = returns.rolling(window=20).std()
    vol_percentile = volatility_20d.rolling(window=50).rank(pct=True)
    
    high_vol_regime = (vol_percentile > 0.8).astype(float)
    low_vol_regime = (vol_percentile < 0.2).astype(float)
    
    # Compute Regime-Specific Returns
    regime_adjusted_return = returns.copy()
    regime_adjusted_return[high_vol_regime == 1] *= 2
    regime_adjusted_return[low_vol_regime == 1] *= 0.5
    
    # Adjust for Volume Trend
    def volume_slope(x):
        if len(x) < 2:
            return np.nan
        return stats.linregress(range(len(x)), x).slope
    
    volume_slope_5d = df['volume'].rolling(window=5).apply(volume_slope, raw=True)
    factor3 = regime_adjusted_return * volume_slope_5d
    
    # Amount-Based Price Efficiency
    # Calculate Price Efficiency Ratio
    abs_return = abs(df['close'].pct_change())
    efficiency_ratio = abs_return / df['amount']
    
    # Compute Efficiency Trend
    efficiency_ma_5d = efficiency_ratio.rolling(window=5).mean()
    efficiency_momentum = efficiency_ratio - efficiency_ma_5d
    
    # Combine with Volume Pattern
    volume_ma_20d = df['volume'].rolling(window=20).mean()
    volume_spike = (df['volume'] > 1.5 * volume_ma_20d).astype(float)
    low_volume = (df['volume'] < 0.8 * volume_ma_20d).astype(float)
    
    efficiency_adjusted = efficiency_momentum.copy()
    efficiency_adjusted[volume_spike == 1] *= 2
    efficiency_adjusted[low_volume == 1] *= 0.5
    factor4 = efficiency_adjusted
    
    # Open-Gap Mean Reversion with Volume Filter
    # Calculate Opening Gap
    gap_pct = (df['open'] / df['close'].shift(1) - 1)
    
    # Assess Mean Reversion Potential
    gap_volatility = gap_pct.rolling(window=10).std()
    gap_mean = gap_pct.rolling(window=10).mean()
    gap_zscore = (gap_pct - gap_mean) / gap_volatility
    
    # Apply Volume Filter
    volume_ma_5d_filter = df['volume'].rolling(window=5).mean()
    volume_indicator = (df['volume'] > volume_ma_5d_filter).astype(float)
    
    # Combine Signals
    factor5 = -gap_zscore * volume_indicator
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'factor1': factor1,
        'factor2': factor2,
        'factor3': factor3,
        'factor4': factor4,
        'factor5': factor5
    })
    
    # Z-score normalize each factor and take average
    normalized_factors = factors.apply(lambda x: (x - x.mean()) / x.std())
    final_factor = normalized_factors.mean(axis=1)
    
    return final_factor
