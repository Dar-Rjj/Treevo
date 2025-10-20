import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Volume-Confirmed Momentum Factor that combines short-term momentum with 
    long-term volume confirmation and convergence-divergence analysis.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Short-Term Momentum Component
    # 5-day Exponential Price Momentum
    ewma_close = close.ewm(alpha=0.2, adjust=False).mean()  # decay=0.8 equivalent
    exp_momentum_5d = (ewma_close - ewma_close.shift(5)) / ewma_close.shift(5)
    
    # 10-day Volume-Weighted Price Momentum
    returns_10d = close.pct_change(periods=10)
    volume_ratio = volume / volume.rolling(10).mean()
    vol_weighted_momentum = returns_10d * volume_ratio
    
    # Combine short-term components with exponential weighting
    short_term_momentum = (exp_momentum_5d * 0.6 + vol_weighted_momentum * 0.4).ewm(alpha=0.1, adjust=False).mean()  # decay=0.9
    
    # Long-Term Confirmation Component
    # 20-day Volatility-Scaled Momentum
    returns_20d = close.pct_change(periods=20)
    volatility_20d = close.pct_change().rolling(20).std()
    vol_scaled_momentum = returns_20d / (volatility_20d + 1e-8)
    
    # 50-day Volume Trend Consistency
    def volume_trend_slope(vol_series):
        if len(vol_series) < 50:
            return np.nan
        x = np.arange(len(vol_series))
        slope, _, _, _, _ = linregress(x, vol_series)
        return slope
    
    volume_trend = volume.rolling(50).apply(volume_trend_slope, raw=False)
    
    # Combine long-term components with exponential weighting
    long_term_confirmation = (vol_scaled_momentum * 0.7 + volume_trend * 0.3).ewm(alpha=0.05, adjust=False).mean()  # decay=0.95
    
    # Convergence-Divergence Analysis
    # Short-Long Momentum Convergence
    momentum_convergence = np.sign(short_term_momentum) * np.sign(long_term_confirmation)
    convergence_strength = (short_term_momentum / (long_term_confirmation + 1e-8)).abs()
    
    # Price-Volume Divergence Penalty
    price_trend = close.rolling(20).apply(lambda x: linregress(np.arange(len(x)), x)[0], raw=False)
    volume_trend_20d = volume.rolling(20).apply(lambda x: linregress(np.arange(len(x)), x)[0], raw=False)
    price_volume_divergence = np.sign(price_trend) != np.sign(volume_trend_20d)
    divergence_penalty = 1.0 - (0.3 * price_volume_divergence.astype(float))
    
    # Factor Combination
    base_factor = (short_term_momentum * momentum_convergence * convergence_strength * 
                   long_term_confirmation * divergence_penalty)
    
    # Apply final exponential smoothing
    factor_smoothed = base_factor.ewm(alpha=0.1, adjust=False).mean()
    
    # Robustness Transformations
    # Winsorization at 2.5% and 97.5% percentiles
    lower_bound = factor_smoothed.quantile(0.025)
    upper_bound = factor_smoothed.quantile(0.975)
    factor_winsorized = factor_smoothed.clip(lower=lower_bound, upper=upper_bound)
    
    # Rank transformation preserving sign
    factor_ranked = factor_winsorized.rank(pct=True) * np.sign(factor_winsorized)
    
    # Final normalization
    final_factor = (factor_ranked - factor_ranked.mean()) / factor_ranked.std()
    
    return final_factor
