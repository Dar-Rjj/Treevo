import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Regime-Switching Price-Volume Cointegration Alpha Factor
    Combines cointegration analysis with microstructure regime detection
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Equilibrium Relationship
    # Calculate price and volume changes
    data['price_change'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    # 20-day rolling correlation between price changes and volume changes
    data['price_volume_corr'] = data['price_change'].rolling(window=20).corr(data['volume_change'])
    
    # Rolling regression residuals as equilibrium deviation
    def rolling_regression_residuals(window):
        if len(window) < 2 or window.isna().any():
            return np.nan
        try:
            X = np.arange(len(window)).reshape(-1, 1)
            y = window.values
            slope, intercept, _, _, _ = stats.linregress(np.arange(len(window)), y)
            predicted = intercept + slope * (len(window) - 1)
            actual = y[-1]
            return actual - predicted
        except:
            return np.nan
    
    # Calculate equilibrium deviation for price and volume separately
    data['price_equilibrium_dev'] = data['close'].rolling(window=20).apply(
        rolling_regression_residuals, raw=False
    )
    data['volume_equilibrium_dev'] = data['volume'].rolling(window=20).apply(
        rolling_regression_residuals, raw=False
    )
    
    # Combined equilibrium deviation
    data['combined_equilibrium_dev'] = (
        data['price_equilibrium_dev'] * data['volume_equilibrium_dev']
    )
    
    # Regime Detection - Market Microstructure
    # Bid-ask spread proxy
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    
    # 10-day rolling variance of spread proxy
    data['spread_variance'] = data['spread_proxy'].rolling(window=10).var()
    
    # Identify high microstructure noise periods
    spread_median = data['spread_variance'].rolling(window=50).median()
    data['high_noise_regime'] = (data['spread_variance'] > spread_median * 1.5).astype(int)
    
    # Correlation stability measure
    data['corr_stability'] = data['price_volume_corr'].rolling(window=10).std()
    corr_stability_median = data['corr_stability'].rolling(window=50).median()
    data['unstable_corr_regime'] = (data['corr_stability'] > corr_stability_median * 1.2).astype(int)
    
    # Regime classification
    # 0: Low noise, stable correlation (strong regime)
    # 1: High noise or unstable correlation (weak regime)
    data['regime_strength'] = 1 - ((data['high_noise_regime'] == 1) | (data['unstable_corr_regime'] == 1)).astype(int)
    
    # Regime confidence based on recent regime persistence
    data['regime_persistence'] = data['regime_strength'].rolling(window=5).mean()
    
    # Generate Regime-Adaptive Alpha Signal
    # Base signal from equilibrium deviation
    base_signal = data['combined_equilibrium_dev']
    
    # Normalize base signal
    base_signal_rolling_mean = base_signal.rolling(window=50).mean()
    base_signal_rolling_std = base_signal.rolling(window=50).std()
    normalized_signal = (base_signal - base_signal_rolling_mean) / base_signal_rolling_std
    
    # Apply regime-based weighting
    # Strong signals in low-noise, high-correlation regimes
    regime_weight = data['regime_strength'] * data['regime_persistence']
    
    # Correlation magnitude weighting
    corr_weight = np.abs(data['price_volume_corr'])
    
    # Final alpha factor
    alpha_factor = normalized_signal * regime_weight * corr_weight
    
    # Clean up any infinite values
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    
    return alpha_factor
