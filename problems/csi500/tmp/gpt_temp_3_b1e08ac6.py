import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining price pattern anomalies, volume dynamics, 
    and market microstructure signals with multi-scale integration.
    """
    data = df.copy()
    
    # Price Pattern Anomalies
    # Fractal price structure breaks using High, Low, Close
    high_rolling = data['high'].rolling(window=5, min_periods=3)
    low_rolling = data['low'].rolling(window=5, min_periods=3)
    close_rolling = data['close'].rolling(window=5, min_periods=3)
    
    fractal_break = (
        (data['high'] > high_rolling.mean() + 1.5 * high_rolling.std()) |
        (data['low'] < low_rolling.mean() - 1.5 * low_rolling.std())
    ).astype(int) * np.sign(data['close'] - close_rolling.mean())
    
    # Asymmetric volatility clustering using High, Low
    high_vol = data['high'].pct_change().abs().rolling(window=10, min_periods=5).std()
    low_vol = data['low'].pct_change().abs().rolling(window=10, min_periods=5).std()
    vol_clustering = (high_vol - low_vol) / (high_vol + low_vol + 1e-8)
    
    # Overnight gap mean reversion using Open, Close
    overnight_gap = (data['open'] / data['close'].shift(1) - 1)
    gap_reversion = -overnight_gap.rolling(window=10, min_periods=5).mean()
    
    # Volume Dynamics Analysis
    # Volume burst autocorrelation
    volume_change = data['volume'].pct_change()
    volume_autocorr = volume_change.rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Large trade concentration shifts (Amount, Volume)
    avg_trade_size = data['amount'] / (data['volume'] + 1e-8)
    trade_concentration = (avg_trade_size - avg_trade_size.rolling(window=20, min_periods=10).mean()) / avg_trade_size.rolling(window=20, min_periods=10).std()
    
    # Volume-weighted price inertia
    volume_weighted_return = (data['close'].pct_change() * data['volume']).rolling(window=5, min_periods=3).sum()
    total_volume = data['volume'].rolling(window=5, min_periods=3).sum()
    price_inertia = volume_weighted_return / (total_volume + 1e-8)
    
    # Market Microstructure Signals
    # Bid-ask spread approximation (High, Low, Close)
    spread_approx = (data['high'] - data['low']) / data['close']
    normalized_spread = (spread_approx - spread_approx.rolling(window=20, min_periods=10).mean()) / spread_approx.rolling(window=20, min_periods=10).std()
    
    # Trade size polarization effects (Amount, Volume)
    trade_size_skew = avg_trade_size.rolling(window=15, min_periods=8).apply(
        lambda x: x.skew() if len(x) > 2 else 0, raw=False
    )
    
    # Price impact asymmetry
    price_range = data['high'] - data['low']
    up_move = (data['close'] - data['open']).clip(lower=0)
    down_move = (data['open'] - data['close']).clip(lower=0)
    
    up_impact = up_move * data['volume'] / (price_range + 1e-8)
    down_impact = down_move * data['volume'] / (price_range + 1e-8)
    
    impact_asymmetry = (up_impact.rolling(window=10, min_periods=5).mean() - 
                       down_impact.rolling(window=10, min_periods=5).mean())
    
    # Factor Integration Framework
    # Cross-sectional regime weighting (using volatility regime)
    volatility_regime = data['close'].pct_change().abs().rolling(window=20, min_periods=10).std()
    high_vol_regime = (volatility_regime > volatility_regime.rolling(window=50, min_periods=25).quantile(0.7)).astype(int)
    
    # Temporal persistence scoring
    price_trend = data['close'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
    )
    
    # Multi-scale signal combination
    short_term = (
        0.3 * fractal_break.rolling(window=3).mean() +
        0.4 * volume_autocorr +
        0.3 * price_inertia
    )
    
    medium_term = (
        0.25 * vol_clustering +
        0.25 * gap_reversion +
        0.25 * trade_concentration +
        0.25 * normalized_spread
    )
    
    long_term = (
        0.4 * trade_size_skew +
        0.4 * impact_asymmetry +
        0.2 * price_trend
    )
    
    # Regime-adjusted final factor
    regime_weight = 0.6 * high_vol_regime + 0.4 * (1 - high_vol_regime)
    
    final_factor = (
        regime_weight * (0.4 * short_term + 0.4 * medium_term + 0.2 * long_term) +
        (1 - regime_weight) * (0.2 * short_term + 0.5 * medium_term + 0.3 * long_term)
    )
    
    # Normalize the final factor
    final_factor = (final_factor - final_factor.rolling(window=50, min_periods=25).mean()) / final_factor.rolling(window=50, min_periods=25).std()
    
    return final_factor
