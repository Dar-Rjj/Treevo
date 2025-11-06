import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Price-Volume Divergence Momentum
    # Short-term Price Momentum
    ret_5d = df['close'] / df['close'].shift(5) - 1
    ret_10d = df['close'] / df['close'].shift(10) - 1
    
    # Volume Trend Persistence
    def volume_slope(series, window=5):
        x = np.arange(window)
        slopes = []
        for i in range(len(series) - window + 1):
            y = series.iloc[i:i+window].values
            if len(y) == window:
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=series.index[window-1:])
    
    vol_slope_5d = volume_slope(df['volume'], window=5)
    vol_accel = vol_slope_5d.diff()
    
    # Divergence Signal
    price_momentum = (ret_5d + ret_10d) / 2
    vol_trend = vol_slope_5d
    divergence_signal = np.where(
        (price_momentum > 0) & (vol_trend < 0) | (price_momentum < 0) & (vol_trend > 0),
        price_momentum * vol_trend.abs(), 0
    )
    
    # High-Low Range Efficiency Ratio
    # True Range Calculation
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_5d = true_range.rolling(5).mean()
    
    # Price Movement Efficiency
    abs_return = abs(df['close'] - df['close'].shift(1))
    efficiency_ratio = abs_return / true_range
    efficiency_ratio = efficiency_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Signal Generation
    efficiency_signal = efficiency_ratio.rolling(5).mean()
    
    # Volume-Scaled Extreme Reversal
    # Extreme Move Identification
    returns = df['close'].pct_change()
    zscore_20d = returns.rolling(20).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False)
    extreme_moves = abs(zscore_20d) > 2
    
    # Volume Context
    median_vol_20d = df['volume'].rolling(20).median()
    volume_ratio = df['volume'] / median_vol_20d
    
    # Reversal Signal
    reversal_signal = np.where(
        extreme_moves & (volume_ratio > 1.5),
        -returns * volume_ratio, 0
    )
    
    # Amount-Based Flow Persistence
    # Daily Flow Direction
    amount_change = df['amount'] - df['amount'].shift(1)
    price_change = df['close'] - df['close'].shift(1)
    
    up_flow = (amount_change > 0) & (price_change > 0)
    down_flow = (amount_change < 0) & (price_change < 0)
    
    # Flow Momentum
    def consecutive_count(series):
        counts = []
        current_count = 0
        for val in series:
            if val:
                current_count += 1
            else:
                current_count = 0
            counts.append(current_count)
        return pd.Series(counts, index=series.index)
    
    up_consecutive = consecutive_count(up_flow)
    down_consecutive = consecutive_count(down_flow)
    flow_momentum = up_consecutive - down_consecutive
    flow_accel = flow_momentum.diff()
    
    # Signal Generation
    flow_signal = flow_momentum * flow_accel
    
    # Volatility-Regime Volume Clustering
    # Volatility Regime Detection
    vol_20d = returns.rolling(20).std()
    vol_regime = vol_20d > vol_20d.median()
    
    # Volume Spike Analysis
    def volume_zscore_by_regime(volume, regime):
        zscores = []
        for i in range(len(volume)):
            if i >= 20:
                current_regime = regime.iloc[i]
                regime_mask = regime.iloc[max(0, i-19):i+1] == current_regime
                regime_volumes = volume.iloc[max(0, i-19):i+1][regime_mask]
                if len(regime_volumes) > 1:
                    zscore = (volume.iloc[i] - regime_volumes.mean()) / regime_volumes.std()
                    zscores.append(zscore)
                else:
                    zscores.append(0)
            else:
                zscores.append(0)
        return pd.Series(zscores, index=volume.index)
    
    vol_zscore = volume_zscore_by_regime(df['volume'], vol_regime)
    volume_spikes = vol_zscore > 2
    
    # Volume clustering detection
    spike_clusters = volume_spikes.rolling(3).sum() >= 2
    
    # Signal Generation
    volatility_signal = np.where(
        spike_clusters & ~vol_regime, -1,  # Low vol clustering: bearish
        np.where(spike_clusters & vol_regime, 1, 0)  # High vol clustering: bullish
    )
    
    # Combine all signals with weights
    combined_signal = (
        0.25 * pd.Series(divergence_signal, index=df.index) +
        0.20 * efficiency_signal +
        0.25 * pd.Series(reversal_signal, index=df.index) +
        0.15 * flow_signal +
        0.15 * pd.Series(volatility_signal, index=df.index)
    )
    
    return combined_signal
