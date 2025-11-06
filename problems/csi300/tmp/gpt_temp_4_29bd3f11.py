import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Dynamic Regime-Adaptive Alpha Factor combining multiple sophisticated approaches
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Market Microstructure Regime Detection
    # Intraday volatility clustering using 5-period rolling window
    returns = data['close'].pct_change()
    vol_5min = returns.rolling(window=5, min_periods=3).std()
    
    # Volatility regime classification
    vol_regime = pd.Series(np.where(vol_5min > vol_5min.rolling(50).quantile(0.7), 2, 
                                   np.where(vol_5min < vol_5min.rolling(50).quantile(0.3), 0, 1)), 
                          index=data.index)
    
    # Liquidity state using volume-weighted price impact
    price_impact = (data['high'] - data['low']) / data['close'] * data['volume']
    liquidity_state = pd.Series(np.where(price_impact > price_impact.rolling(20).quantile(0.7), 0,
                                       np.where(price_impact < price_impact.rolling(20).quantile(0.3), 2, 1)),
                              index=data.index)
    
    # 2. Quantum-Inspired Price-Volume Entanglement
    # Phase synchronization between price and volume
    price_change = data['close'].pct_change()
    volume_change = data['volume'].pct_change()
    
    # Quantum correlation using rolling correlation with phase adjustment
    quantum_corr = price_change.rolling(window=10).corr(volume_change)
    entanglement_strength = quantum_corr.abs()
    
    # Quantum tunneling detection - penetration of rolling high/low
    rolling_high = data['high'].rolling(window=20).max()
    rolling_low = data['low'].rolling(window=20).min()
    tunneling = ((data['close'] > rolling_high.shift(1)) | (data['close'] < rolling_low.shift(1))).astype(int)
    
    # 3. Fractal Market Microstructure Analysis
    # Multi-scale Hurst exponent approximation
    def hurst_approx(series, max_lag=10):
        lags = range(2, max_lag+1)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    hurst_20 = data['close'].rolling(window=50).apply(hurst_approx, raw=False)
    fractal_regime = pd.Series(np.where(hurst_20 > 0.6, 1, np.where(hurst_20 < 0.4, -1, 0)), 
                             index=data.index)
    
    # 4. Neuromorphic Market Memory Factor
    # Short-term memory (recent price patterns)
    price_memory = data['close'].rolling(window=5).apply(
        lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 3 else 0, raw=False
    )
    
    # Volume pattern memory
    volume_memory = data['volume'].rolling(window=5).apply(
        lambda x: np.std(x) / np.mean(x) if np.mean(x) > 0 else 0, raw=False
    )
    
    # Memory consistency (pattern persistence)
    memory_consistency = (price_memory.rolling(window=10).std() / 
                         price_memory.rolling(window=10).mean()).replace([np.inf, -np.inf], 0).fillna(0)
    
    # 5. Adaptive Factor Integration
    # Regime-specific momentum factors
    # High volatility: mean-reverting momentum (short lookback)
    mom_short = data['close'].pct_change(periods=3)
    
    # Low volatility: trend-following momentum (medium lookback)
    mom_medium = data['close'].pct_change(periods=10)
    
    # Regime-adaptive momentum
    regime_momentum = pd.Series(np.where(vol_regime == 2, mom_short,
                                       np.where(vol_regime == 0, mom_medium,
                                               (mom_short + mom_medium) / 2)), 
                              index=data.index)
    
    # Microstructure-informed volume factor
    volume_effectiveness = (data['close'].pct_change().abs() / 
                          (data['volume'] / data['volume'].rolling(20).mean())).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Regime-weighted volume factor
    regime_volume = pd.Series(np.where(liquidity_state == 0, 
                                     volume_effectiveness.rolling(5).std(),
                                     np.where(liquidity_state == 2,
                                             volume_effectiveness.rolling(5).mean(),
                                             volume_effectiveness)), 
                            index=data.index)
    
    # 6. Final Alpha Factor Combination
    # Entanglement-enhanced momentum
    entanglement_momentum = regime_momentum * (1 + entanglement_strength)
    
    # Fractal-adaptive signals
    fractal_signal = pd.Series(np.where(fractal_regime == 1, regime_momentum,
                                      np.where(fractal_regime == -1, -regime_momentum,
                                              regime_momentum * 0.5)), 
                             index=data.index)
    
    # Memory-weighted factors
    memory_weight = 1 / (1 + memory_consistency)
    memory_enhanced = (entanglement_momentum + fractal_signal) * memory_weight
    
    # Quantum tunneling boost
    tunneling_boost = tunneling * np.sign(regime_momentum) * 0.2
    
    # Final alpha factor with regime adjustments
    alpha = (memory_enhanced * 0.6 + 
             regime_volume * 0.3 + 
             tunneling_boost * 0.1)
    
    # Volatility normalization
    alpha_vol = alpha.rolling(window=20).std()
    normalized_alpha = alpha / alpha_vol.replace(0, 1)
    
    return normalized_alpha.fillna(0)
