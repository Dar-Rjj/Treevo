import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Price-Volume Divergence Momentum with Volatility-Adjusted Persistence
    and Efficiency-Weighted Alignment
    """
    # Price Trend Component
    # Short-term Price Momentum
    ret_5d = df['close'] / df['close'].shift(5) - 1
    ret_10d = df['close'] / df['close'].shift(10) - 1
    
    # Medium-term Price Trend
    def price_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return stats.linregress(x, series.values)[0]
    
    price_slope_20d = df['close'].rolling(window=20).apply(price_slope, raw=False)
    
    # Price Volatility Adjustment (ATR)
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    atr_20d = tr.rolling(window=20).mean()
    price_trend = price_slope_20d / atr_20d
    
    # Volume Trend Component
    # Volume Momentum Analysis
    vol_change_5d = df['volume'] / df['volume'].shift(5) - 1
    
    def volume_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return stats.linregress(x, series.values)[0]
    
    vol_slope_10d = df['volume'].rolling(window=10).apply(volume_slope, raw=False)
    
    # Volume-Price Correlation
    daily_returns = df['close'].pct_change()
    daily_vol_changes = df['volume'].pct_change()
    vol_price_corr_20d = daily_returns.rolling(window=20).corr(daily_vol_changes)
    
    # Multi-timeframe Convergence Factor
    # Short-term Convergence (5-day)
    price_momentum_5d = (df['close'] / df['close'].shift(5) - 1)
    vol_momentum_5d = (df['volume'] / df['volume'].shift(5) - 1)
    
    # Medium-term Convergence (20-day)
    price_trend_strength = price_slope_20d / price_slope_20d.rolling(window=20).std()
    vol_trend_strength = vol_slope_10d / vol_slope_10d.rolling(window=20).std()
    
    # Convergence Signal Generation
    short_term_align = np.sign(price_momentum_5d) * np.sign(vol_momentum_5d)
    medium_term_align = np.sign(price_trend_strength) * np.sign(vol_trend_strength)
    convergence_score = (0.4 * short_term_align + 0.6 * medium_term_align)
    
    # Volatility-Adjusted Momentum Persistence
    # Multi-period Return Calculation
    returns_1d = df['close'].pct_change()
    returns_5d = df['close'].pct_change(5)
    
    # Return Autocorrelation Analysis
    autocorr_1d = returns_1d.rolling(window=20).corr(returns_1d.shift(1))
    autocorr_5d = returns_5d.rolling(window=20).corr(returns_5d.shift(5))
    
    # Volatility Normalization
    daily_ranges = (df['high'] - df['low']) / df['close']
    range_vol_20d = daily_ranges.rolling(window=20).std()
    return_vol_20d = returns_1d.rolling(window=20).std()
    
    # Risk-Adjusted Return Measures
    sharpe_5d = returns_5d / return_vol_20d
    vol_scaled_momentum = price_momentum_5d / range_vol_20d
    
    # Persistence Detection
    positive_momentum_streak = (returns_1d > 0).rolling(window=5).sum()
    vol_confirmed_persistence = positive_momentum_streak * vol_momentum_5d
    
    # Regime-based Factor Weighting
    vol_regime = (range_vol_20d > range_vol_20d.rolling(window=60).median()).astype(int)
    regime_weight = 0.7 + 0.3 * vol_regime  # Higher weight in high vol regimes
    
    # Efficiency-Weighted Price-Volume Alignment
    # Movement Efficiency Analysis
    close_movement = abs(df['close'].pct_change())
    intraday_range = (df['high'] - df['low']) / df['close']
    price_efficiency = close_movement / intraday_range.replace(0, np.nan)
    
    # Volume Efficiency
    volume_per_move = df['volume'] / (abs(returns_1d).replace(0, np.nan) * df['close'])
    volume_range_efficiency = df['volume'] / (intraday_range.replace(0, np.nan) * df['close'])
    
    # Multi-timeframe Efficiency Convergence
    eff_momentum_5d = price_efficiency.rolling(window=5).mean() / price_efficiency.rolling(window=20).mean() - 1
    
    # Integrated Factor Generation
    composite_efficiency = price_efficiency * volume_range_efficiency
    
    # Final Alpha Factor
    price_volume_divergence = convergence_score * (1 - abs(vol_price_corr_20d))
    volatility_adjusted = vol_scaled_momentum * regime_weight
    efficiency_weighted = composite_efficiency * eff_momentum_5d
    
    # Combine all components
    alpha_factor = (
        0.35 * price_volume_divergence +
        0.40 * volatility_adjusted +
        0.25 * efficiency_weighted
    )
    
    return alpha_factor
