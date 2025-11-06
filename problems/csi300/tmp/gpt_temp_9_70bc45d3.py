import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility regime adjusted momentum,
    price-volume divergence convergence, and liquidity-adjusted reversal.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Adjusted Momentum
    # Calculate 5-day momentum
    momentum_5d = data['close'].pct_change(5)
    
    # Calculate 20-day average true range
    tr = np.maximum(data['high'] - data['low'], 
                   np.maximum(abs(data['high'] - data['close'].shift(1)), 
                             abs(data['low'] - data['close'].shift(1))))
    atr_20d = tr.rolling(window=20, min_periods=10).mean()
    
    # Calculate volatility autocorrelation (regime persistence)
    vol_autocorr = atr_20d.rolling(window=20, min_periods=10).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Adjust momentum by volatility regime
    vol_adjusted_momentum = momentum_5d / (atr_20d + 1e-8)
    regime_weighted_momentum = vol_adjusted_momentum * (1 + vol_autocorr)
    
    # Price-Volume Divergence Convergence
    # Calculate 10-day price trend slope
    def linear_slope(x):
        if len(x) < 2:
            return 0
        return stats.linregress(range(len(x)), x)[0]
    
    price_trend = data['close'].rolling(window=10, min_periods=5).apply(
        linear_slope, raw=False
    )
    
    # Calculate 10-day volume trend slope
    volume_trend = data['volume'].rolling(window=10, min_periods=5).apply(
        linear_slope, raw=False
    )
    
    # Calculate rolling correlation between price and volume trends
    def rolling_corr(x):
        if len(x) < 2:
            return 0
        price_vals = x[:, 0]
        volume_vals = x[:, 1]
        if len(price_vals) < 2 or len(volume_vals) < 2:
            return 0
        return np.corrcoef(price_vals, volume_vals)[0, 1] if not np.isnan(np.corrcoef(price_vals, volume_vals)[0, 1]) else 0
    
    # Create combined array for correlation calculation
    combined_data = np.column_stack([price_trend, volume_trend])
    trend_correlation = pd.Series(index=data.index, dtype=float)
    
    for i in range(10, len(data)):
        if i >= 10:
            window_data = combined_data[i-9:i+1]
            trend_correlation.iloc[i] = rolling_corr(window_data)
    
    # Calculate convergence speed (rate of correlation change)
    convergence_speed = trend_correlation.diff(3)
    
    # Price-volume divergence signal (negative correlation suggests divergence)
    pv_divergence = -trend_correlation * convergence_speed
    
    # Liquidity-Adjusted Reversal
    # Calculate 1-day reversal
    reversal_1d = -data['close'].pct_change(1)
    
    # Calculate effective spread proxy (amount per volume)
    effective_spread = (data['amount'] / (data['volume'] + 1e-8)).rolling(
        window=5, min_periods=3
    ).mean()
    
    # Calculate liquidity persistence (autocorrelation of volume)
    volume_autocorr = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Apply liquidity filter to reversal
    liquidity_weight = 1 / (effective_spread + 1e-8)
    liquidity_adjusted_reversal = reversal_1d * liquidity_weight * (1 + volume_autocorr)
    
    # Combine all components with equal weighting
    alpha_factor = (
        regime_weighted_momentum.fillna(0) * 0.4 +
        pv_divergence.fillna(0) * 0.3 +
        liquidity_adjusted_reversal.fillna(0) * 0.3
    )
    
    # Normalize the factor
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20, min_periods=10).mean()) / (
        alpha_factor.rolling(window=20, min_periods=10).std() + 1e-8
    )
    
    return alpha_factor
