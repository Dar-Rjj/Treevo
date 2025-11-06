import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import entropy

def heuristics_v2(data):
    """
    Multi-Scale Volatility Regime Entropy Factor
    Combines volatility clustering persistence, entropy of price changes, 
    regime detection, and stability weighting
    """
    df = data.copy()
    
    # Compute multi-scale volatility (5, 10, 20 days)
    returns = df['close'].pct_change()
    
    # Volatility clustering persistence
    vol_5 = returns.rolling(window=5).std()
    vol_10 = returns.rolling(window=10).std()
    vol_20 = returns.rolling(window=20).std()
    
    # Volatility persistence score (autocorrelation of volatility)
    vol_persistence = (vol_5.rolling(window=10).corr(vol_5.shift(1)) + 
                      vol_10.rolling(window=10).corr(vol_10.shift(1)) + 
                      vol_20.rolling(window=10).corr(vol_20.shift(1))) / 3
    
    # Calculate entropy of price change magnitude distributions
    price_changes = np.abs(returns)
    
    # Create bins for entropy calculation (10 bins over rolling window)
    def rolling_entropy(series, window=20):
        entropy_vals = []
        for i in range(len(series)):
            if i < window:
                entropy_vals.append(np.nan)
            else:
                window_data = series.iloc[i-window:i]
                hist, _ = np.histogram(window_data.dropna(), bins=10, density=True)
                hist = hist[hist > 0]  # Remove zeros for log calculation
                if len(hist) > 1:
                    entropy_vals.append(entropy(hist))
                else:
                    entropy_vals.append(np.nan)
        return pd.Series(entropy_vals, index=series.index)
    
    price_entropy = rolling_entropy(price_changes, window=20)
    
    # Regime detection using volatility state models
    vol_ratio = vol_5 / vol_20
    high_vol_regime = (vol_ratio > 1.2).astype(int)
    low_vol_regime = (vol_ratio < 0.8).astype(int)
    
    # Regime transitions
    regime_transitions = (high_vol_regime.diff().abs() + low_vol_regime.diff().abs())
    
    # Regime stability (persistence of current regime)
    regime_stability = high_vol_regime.rolling(window=10).sum() + low_vol_regime.rolling(window=10).sum()
    
    # Transition magnitude (volatility change during transitions)
    transition_magnitude = vol_ratio.diff().abs() * regime_transitions
    
    # Combine components with weights
    factor = (vol_persistence * 0.3 + 
              (1 - price_entropy / price_entropy.rolling(window=50).max()) * 0.4 +
              regime_stability * 0.2 - 
              transition_magnitude * 0.1)
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=50).mean()) / factor.rolling(window=50).std()
    
    return factor
