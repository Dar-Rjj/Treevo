import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining volatility regime momentum, 
    volume-weighted price efficiency, cross-timeframe divergence,
    liquidity-driven reversals, and trend-following regime switching.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    # Volatility Regime Momentum
    # Parkinson volatility estimator
    hl_range = np.log(df['high'] / df['low'])
    realized_vol = (hl_range ** 2) / (4 * np.log(2))
    
    # Classify volatility states using rolling percentiles
    vol_20d = realized_vol.rolling(window=20, min_periods=10).mean()
    vol_low_thresh = vol_20d.rolling(window=60, min_periods=30).quantile(0.3)
    vol_high_thresh = vol_20d.rolling(window=60, min_periods=30).quantile(0.7)
    
    vol_regime = pd.Series(index=df.index, dtype=str)
    vol_regime[:] = 'medium'
    vol_regime[vol_20d < vol_low_thresh] = 'low'
    vol_regime[vol_20d > vol_high_thresh] = 'high'
    
    # Momentum by regime
    returns_5d = df['close'].pct_change(5)
    momentum_low = returns_5d.rolling(window=10, min_periods=5).mean()
    momentum_medium = returns_5d.rolling(window=5, min_periods=3).mean()
    momentum_high = returns_5d.rolling(window=3, min_periods=2).mean()
    
    regime_momentum = pd.Series(index=df.index, dtype=float)
    regime_momentum[vol_regime == 'low'] = momentum_low[vol_regime == 'low']
    regime_momentum[vol_regime == 'medium'] = momentum_medium[vol_regime == 'medium']
    regime_momentum[vol_regime == 'high'] = momentum_high[vol_regime == 'high']
    
    # Volume-Weighted Price Efficiency
    # Variance ratio across multiple horizons
    def variance_ratio(series, horizon):
        if len(series) < horizon + 1:
            return np.nan
        returns = series.pct_change().dropna()
        var_1 = returns.var()
        var_h = returns.rolling(window=horizon).sum().var()
        return var_h / (horizon * var_1) if var_1 != 0 else 1.0
    
    vr_5d = pd.Series([variance_ratio(df['close'].iloc[:i+1], 5) if i >= 5 else np.nan 
                      for i in range(len(df))], index=df.index)
    vr_10d = pd.Series([variance_ratio(df['close'].iloc[:i+1], 10) if i >= 10 else np.nan 
                       for i in range(len(df))], index=df.index)
    
    price_efficiency = (vr_5d + vr_10d) / 2
    volume_weight = df['volume'].rolling(window=10, min_periods=5).mean()
    weighted_efficiency = price_efficiency * volume_weight
    
    # Cross-Timeframe Divergence
    intraday_move = np.log(df['high'] / df['low'])
    overnight_move = np.log(df['open'] / df['close'].shift(1)).abs()
    
    divergence = intraday_move - overnight_move.rolling(window=5, min_periods=3).mean()
    volume_confirmation = df['volume'] / df['volume'].rolling(window=10, min_periods=5).mean()
    divergence_signal = divergence * volume_confirmation
    
    # Price range expansion scaling
    range_expansion = df['high'] - df['low']
    range_ratio = range_expansion / range_expansion.rolling(window=10, min_periods=5).mean()
    scaled_divergence = divergence_signal * range_ratio
    
    # Liquidity-Driven Reversals
    # Bid-ask spread proxy (using daily range normalized by price)
    spread_proxy = (df['high'] - df['low']) / df['close']
    
    # Price impact per unit volume
    price_impact = (df['close'] - df['open']).abs() / (df['volume'] + 1e-8)
    
    # Volume-spike reversals
    volume_spike = df['volume'] / df['volume'].rolling(window=20, min_periods=10).mean()
    reversal_strength = -df['close'].pct_change(3) * volume_spike
    
    # Local liquidity conditions
    liquidity_conditions = 1 / (spread_proxy.rolling(window=10, min_periods=5).mean() + 1e-8)
    weighted_reversal = reversal_strength * liquidity_conditions
    
    # Trend-Following Regime Switching
    # Multi-scale moving averages
    ma_short = df['close'].rolling(window=5, min_periods=3).mean()
    ma_medium = df['close'].rolling(window=10, min_periods=5).mean()
    ma_long = df['close'].rolling(window=20, min_periods=10).mean()
    
    # Trend transitions
    trend_up = (ma_short > ma_medium) & (ma_medium > ma_long)
    trend_down = (ma_short < ma_medium) & (ma_medium < ma_long)
    
    # Volume-trend alignment
    volume_trend_alignment = (df['volume'] / df['volume'].rolling(window=10, min_periods=5).mean()) * \
                            (df['close'].pct_change(1).abs())
    
    # Regime stability (price oscillation)
    price_oscillation = (df['high'] - df['low']) / df['close']
    regime_stability = 1 / (price_oscillation.rolling(window=10, min_periods=5).std() + 1e-8)
    
    # Combine all components with appropriate weights
    alpha_factor = (
        0.25 * regime_momentum +
        0.20 * weighted_efficiency +
        0.20 * scaled_divergence +
        0.20 * weighted_reversal +
        0.15 * (trend_up.astype(float) - trend_down.astype(float)) * 
        volume_trend_alignment * regime_stability
    )
    
    # Normalize the final factor
    result = (alpha_factor - alpha_factor.rolling(window=20, min_periods=10).mean()) / \
             (alpha_factor.rolling(window=20, min_periods=10).std() + 1e-8)
    
    return result
