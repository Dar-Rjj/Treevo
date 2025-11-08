import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum-Volume Divergence with Exponential Smoothing
    """
    df = data.copy()
    
    # Momentum Components Calculation
    # Price Momentum Series
    df['price_momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['price_momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # Volume Momentum Series
    df['volume_momentum_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_momentum_10d'] = df['volume'] / df['volume'].shift(10) - 1
    df['volume_momentum_20d'] = df['volume'] / df['volume'].shift(20) - 1
    
    # Regime Detection Framework
    # Amount-Based Regime Classification
    df['amount_momentum_5d'] = df['amount'] / df['amount'].shift(5) - 1
    df['amount_acceleration'] = df['amount_momentum_5d'] - (df['amount'].shift(5) / df['amount'].shift(10) - 1)
    df['regime_transition_indicator'] = np.abs(df['amount_acceleration']) * df['amount_momentum_5d']
    
    # Volatility Regime Assessment
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['avg_range_10d'] = df['daily_range'].rolling(window=10).mean()
    df['volatility_regime'] = df['daily_range'] / df['avg_range_10d']
    
    # Exponential Smoothing Application
    # Price Momentum Smoothing
    alpha = 0.3
    df['ema_5d_price'] = df['price_momentum_5d'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_10d_price'] = df['price_momentum_10d'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_20d_price'] = df['price_momentum_20d'].ewm(alpha=alpha, adjust=False).mean()
    
    # Volume Momentum Smoothing
    df['ema_5d_volume'] = df['volume_momentum_5d'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_10d_volume'] = df['volume_momentum_10d'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_20d_volume'] = df['volume_momentum_20d'].ewm(alpha=alpha, adjust=False).mean()
    
    # Momentum-Volume Divergence Calculation
    df['short_term_divergence'] = df['ema_5d_price'] - df['ema_5d_volume']
    df['medium_term_divergence'] = df['ema_10d_price'] - df['ema_10d_volume']
    df['long_term_divergence'] = df['ema_20d_price'] - df['ema_20d_volume']
    
    # Regime-Adaptive Weighting
    regime_weighted_divergence = np.zeros(len(df))
    
    # High Volatility Regime
    high_vol_mask = df['volatility_regime'] > 1.2
    regime_weighted_divergence[high_vol_mask] = (
        0.6 * df['short_term_divergence'] + 0.4 * df['medium_term_divergence']
    )[high_vol_mask] * (1 + np.abs(df['regime_transition_indicator'][high_vol_mask]))
    
    # Low Volatility Regime
    low_vol_mask = df['volatility_regime'] < 0.8
    regime_weighted_divergence[low_vol_mask] = (
        0.4 * df['short_term_divergence'] + 0.6 * df['medium_term_divergence']
    )[low_vol_mask] * (1 - np.abs(df['regime_transition_indicator'][low_vol_mask]))
    
    # Normal Regime
    normal_vol_mask = (df['volatility_regime'] >= 0.8) & (df['volatility_regime'] <= 1.2)
    regime_weighted_divergence[normal_vol_mask] = (
        0.5 * df['short_term_divergence'] + 0.5 * df['medium_term_divergence']
    )[normal_vol_mask]
    
    df['regime_weighted_divergence'] = regime_weighted_divergence
    
    # Cross-Sectional Ranking
    # For single stock context, use rolling percentiles for universe-relative positioning
    df['divergence_rank'] = df['regime_weighted_divergence'].rolling(window=252, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df['long_term_rank'] = df['long_term_divergence'].rolling(window=252, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df['combined_rank'] = (df['divergence_rank'] + df['long_term_rank']) / 2
    
    # Outlier detection and adjustment
    regime_median = df['regime_transition_indicator'].rolling(window=252, min_periods=1).median()
    outlier_mask = df['regime_transition_indicator'] > (2 * regime_median)
    
    # Final Alpha Factor Construction
    base_signal = df['regime_weighted_divergence']
    
    # Apply cross-sectional adjustment
    cross_sectional_adjustment = df['combined_rank']
    
    # Apply regime transition enhancement
    regime_enhancement = 1 + df['regime_transition_indicator']
    
    # Construct final factor
    final_factor = base_signal * cross_sectional_adjustment * regime_enhancement
    
    # Apply outlier adjustment
    final_factor[outlier_mask] = final_factor[outlier_mask] * 0.5
    
    return final_factor
