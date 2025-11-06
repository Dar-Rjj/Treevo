import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Volatility Regime Adjusted Skewness Factor
    skewness_20d = returns.rolling(window=20).apply(lambda x: stats.skew(x), raw=True)
    volatility_20d = returns.rolling(window=20).std()
    volatility_adjusted_skewness = skewness_20d * volatility_20d
    
    # Momentum Acceleration Decay Factor
    momentum_5d = returns.rolling(window=5).sum()
    momentum_20d = returns.rolling(window=20).sum()
    momentum_acceleration = momentum_5d - momentum_20d
    
    # Apply exponential decay weighting
    decay_weights = np.exp(-np.arange(len(momentum_acceleration)) / 10)
    decay_weights = decay_weights / decay_weights.sum()
    momentum_acceleration_decay = momentum_acceleration.rolling(window=len(decay_weights), min_periods=1).apply(
        lambda x: np.dot(x, decay_weights[:len(x)]), raw=False
    )
    
    # Volume-Price Divergence Oscillator
    def linear_regression_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return np.cov(x, series)[0, 1] / np.var(x)
    
    price_slope_10d = df['close'].rolling(window=10).apply(linear_regression_slope, raw=False)
    volume_slope_10d = df['volume'].rolling(window=10).apply(linear_regression_slope, raw=False)
    volume_price_divergence = price_slope_10d * volume_slope_10d
    
    # Liquidity-Adjusted Reversal Factor
    reversal_1d = -returns  # Negative of 1-day return for reversal
    reversal_5d = -returns.rolling(window=5).sum()  # Negative of 5-day return for reversal
    
    # Combined reversal signal
    combined_reversal = reversal_1d + reversal_5d
    
    # Liquidity proxy (average dollar volume over 5 days)
    dollar_volume = df['close'] * df['volume']
    liquidity_5d = dollar_volume.rolling(window=5).mean()
    
    # Volatility normalization (20-day volatility)
    vol_normalization = returns.rolling(window=20).std()
    
    # Liquidity-adjusted reversal
    liquidity_adjusted_reversal = combined_reversal * liquidity_5d / vol_normalization
    
    # Regime-Switching Mean Reversion
    volatility_20d_ma = returns.rolling(window=20).std()
    historical_vol_median = volatility_20d_ma.expanding().median()
    
    # High volatility periods (volatility above historical median)
    high_vol_regime = (volatility_20d_ma > historical_vol_median).astype(float)
    
    # Mean reversion signal (current price vs 20-day moving average)
    ma_20d = df['close'].rolling(window=20).mean()
    price_deviation = (df['close'] - ma_20d) / ma_20d
    
    # Regime-switching mean reversion
    regime_switching_reversion = price_deviation * (1 + high_vol_regime)
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'vol_adj_skew': volatility_adjusted_skewness,
        'mom_accel_decay': momentum_acceleration_decay,
        'vol_price_div': volume_price_divergence,
        'liq_adj_rev': liquidity_adjusted_reversal,
        'regime_reversion': regime_switching_reversion
    })
    
    # Remove any infinite values and fill NaN with 0
    factors = factors.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.mean()) / x.std())
    
    # Equal-weighted composite factor
    composite_factor = factors_normalized.mean(axis=1)
    
    return composite_factor
