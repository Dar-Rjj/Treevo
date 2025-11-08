import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-normalized momentum, volume divergence,
    adaptive trend strength, and price efficiency momentum.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # 1. Volatility-Normalized Multi-Timeframe Momentum
    # Calculate short-term momentum (3-day)
    mom_short = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    
    # Calculate medium-term momentum (10-day)
    mom_medium = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    # Calculate 20-day rolling volatility
    returns = data['close'].pct_change()
    vol_20d = returns.rolling(window=20).std()
    
    # Normalize momentum components by volatility
    mom_short_norm = mom_short / vol_20d.replace(0, np.nan)
    mom_medium_norm = mom_medium / vol_20d.replace(0, np.nan)
    
    # Apply smoothing and trend consistency weighting
    mom_short_smooth = mom_short_norm.rolling(window=5).mean()
    mom_medium_smooth = mom_medium_norm.rolling(window=5).mean()
    
    # Trend consistency (correlation between short and medium term)
    trend_consistency = mom_short.rolling(window=10).corr(mom_medium)
    
    # Combine momentum components
    momentum_factor = 0.5 * mom_short_smooth + 0.5 * mom_medium_smooth
    momentum_factor = momentum_factor * trend_consistency.fillna(0)
    
    # 2. Volume Divergence Regime Factor
    # Calculate price and volume trend slopes (5-day)
    def calc_slope(series, window=5):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                if len(y) == window and not np.isnan(y).any():
                    slope, _, _, _, _ = stats.linregress(x, y)
                    slopes.iloc[i] = slope
        return slopes
    
    price_slope = calc_slope(data['close'])
    volume_slope = calc_slope(data['volume'])
    
    # Detect divergence patterns
    price_volume_divergence = np.zeros(len(data))
    for i in range(len(data)):
        if i >= 4:  # Ensure we have enough data
            if price_slope.iloc[i] > 0 and volume_slope.iloc[i] < 0:
                price_volume_divergence[i] = -1  # Bearish divergence
            elif price_slope.iloc[i] < 0 and volume_slope.iloc[i] > 0:
                price_volume_divergence[i] = 1   # Bullish divergence
    
    # Volume surge detection
    volume_mean = data['volume'].rolling(window=20).mean()
    volume_std = data['volume'].rolling(window=20).std()
    volume_zscore = (data['volume'] - volume_mean) / volume_std.replace(0, np.nan)
    
    # Combine volume signals
    volume_signal = volume_zscore * price_volume_divergence
    
    # Regime-based weighting
    vol_regime = (vol_20d > vol_20d.rolling(window=60).median()).astype(int)
    # Lower weights in volatile regimes
    regime_weight = 1.0 - 0.3 * vol_regime
    volume_factor = volume_signal * regime_weight
    
    # 3. Adaptive Trend Strength Indicator
    # Multi-scale trend measurement
    ma_3d = data['close'].rolling(window=3).mean()
    ma_10d = data['close'].rolling(window=10).mean()
    
    short_trend = (data['close'] - ma_3d) / ma_3d.replace(0, np.nan)
    medium_trend = (data['close'] - ma_10d) / ma_10d.replace(0, np.nan)
    
    # Trend consistency score
    trend_alignment = (np.sign(short_trend) == np.sign(medium_trend)).astype(float)
    trend_strength = (abs(short_trend) + abs(medium_trend)) / 2
    
    # Volume confirmation
    volume_trend_alignment = np.zeros(len(data))
    for i in range(len(data)):
        if i >= 4:
            if (short_trend.iloc[i] > 0 and volume_slope.iloc[i] > 0) or \
               (short_trend.iloc[i] < 0 and volume_slope.iloc[i] < 0):
                volume_trend_alignment[i] = 1
            else:
                volume_trend_alignment[i] = -1
    
    # Combine trend signals
    trend_factor = trend_strength * trend_alignment * volume_trend_alignment
    
    # Regime-adaptive adjustment
    trend_factor = trend_factor * regime_weight
    
    # 4. Price Efficiency Momentum Factor
    # Intraday efficiency measure
    efficiency_ratio = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    efficiency_ma = efficiency_ratio.rolling(window=5).mean()
    efficiency_momentum = efficiency_ratio - efficiency_ma.shift(1)
    
    # Volume-efficiency relationship
    efficiency_volume_corr = efficiency_ratio.rolling(window=10).corr(data['volume'])
    
    # Combine efficiency signals
    efficiency_signal = efficiency_momentum * efficiency_volume_corr.fillna(0)
    
    # Volatility-adjusted efficiency
    efficiency_factor = efficiency_signal / vol_20d.replace(0, np.nan)
    
    # 5. Final Alpha Factor Combination
    # Normalize all factors to comparable scales
    factors = pd.DataFrame({
        'momentum': momentum_factor,
        'volume': volume_factor,
        'trend': trend_factor,
        'efficiency': efficiency_factor
    })
    
    # Z-score normalization
    factors_normalized = factors.apply(lambda x: (x - x.rolling(window=20).mean()) / x.rolling(window=20).std().replace(0, np.nan))
    
    # Equal weighted combination
    final_factor = factors_normalized.mean(axis=1)
    
    return final_factor
