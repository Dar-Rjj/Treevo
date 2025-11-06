import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Volatility-Normalized Momentum
    # Short-term momentum: (close_t / close_{t-5}) - 1
    momentum = (df['close'] / df['close'].shift(5)) - 1
    
    # Volatility proxy: (high_t - low_t) / close_{t-1}
    volatility_proxy = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Normalized momentum: momentum / volatility_proxy
    normalized_momentum = momentum / volatility_proxy
    
    # Volume-Price Divergence
    def linear_regression_slope(series, window=5):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.iloc[i] = slope
        return slopes
    
    # Volume trend: 5-day linear regression slope of volume
    volume_trend = linear_regression_slope(df['volume'], window=5)
    
    # Price trend: 5-day linear regression slope of close
    price_trend = linear_regression_slope(df['close'], window=5)
    
    # Divergence signal: sign(volume_trend) ≠ sign(price_trend)
    divergence_flag = (np.sign(volume_trend) != np.sign(price_trend)).astype(int)
    
    # Regime Detection
    # Volatility regime: 20-day average of daily ranges
    daily_range = (df['high'] - df['low']) / df['close'].shift(1)
    volatility_regime = daily_range.rolling(window=20).mean()
    
    # High volatility: above median of past 252 days
    vol_median = volatility_regime.rolling(window=252).median()
    is_high_vol = (volatility_regime > vol_median).astype(int)
    
    # Adaptive Signal Combination
    # Base signal: volatility-normalized momentum
    base_signal = normalized_momentum
    
    # Volume-confirmed signal: base_signal × (1 + divergence_flag)
    volume_confirmed_signal = base_signal * (1 + divergence_flag)
    
    # Regime-adjusted alpha
    regime_adjusted_alpha = volume_confirmed_signal.copy()
    regime_adjusted_alpha[is_high_vol == 1] = volume_confirmed_signal[is_high_vol == 1] * 0.7
    regime_adjusted_alpha[is_high_vol == 0] = volume_confirmed_signal[is_high_vol == 0] * 1.3
    
    return regime_adjusted_alpha
