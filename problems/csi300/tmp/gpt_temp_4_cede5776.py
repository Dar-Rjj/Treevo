import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Volume Entropy Regime Momentum factor
    Combines volatility entropy, volume entropy, and momentum dynamics to identify regime transitions
    """
    # Calculate log returns
    df = df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Multi-timeframe volatility entropy (5, 10, 20 days)
    vol_windows = [5, 10, 20]
    vol_entropy_components = []
    
    for window in vol_windows:
        # Calculate rolling volatility (standard deviation of returns)
        rolling_vol = df['log_ret'].rolling(window=window).std()
        
        # Calculate volatility entropy using Shannon entropy on volatility bins
        vol_bins = pd.cut(rolling_vol, bins=10, labels=False)
        vol_prob = vol_bins.value_counts(normalize=True).sort_index()
        vol_entropy = -np.sum(vol_prob * np.log(vol_prob + 1e-10))
        vol_entropy_series = pd.Series(vol_entropy, index=df.index).fillna(method='ffill')
        vol_entropy_components.append(vol_entropy_series)
    
    # Combine multi-timeframe volatility entropy
    vol_entropy_combined = pd.concat(vol_entropy_components, axis=1).mean(axis=1)
    
    # Volume entropy using volume distribution dispersion
    volume_windows = [5, 10, 20]
    volume_entropy_components = []
    
    for window in volume_windows:
        # Calculate volume distribution entropy
        volume_rolling = df['volume'].rolling(window=window)
        volume_bins = pd.cut(df['volume'], bins=10, labels=False)
        volume_prob = volume_bins.rolling(window=window).apply(
            lambda x: -np.sum(pd.Series(x).value_counts(normalize=True) * 
                            np.log(pd.Series(x).value_counts(normalize=True) + 1e-10))
        )
        volume_entropy_components.append(volume_prob)
    
    # Combine volume entropy components
    volume_entropy_combined = pd.concat(volume_entropy_components, axis=1).mean(axis=1)
    
    # Joint volatility-volume entropy regime identification
    joint_entropy = vol_entropy_combined * volume_entropy_combined
    regime_breaks = joint_entropy.rolling(window=10).std() / joint_entropy.rolling(window=10).mean()
    
    # Regime-specific momentum entropy
    momentum_windows = [5, 10, 15]
    momentum_entropy_components = []
    
    for window in momentum_windows:
        # Calculate momentum (simple returns)
        momentum = df['close'].pct_change(window)
        
        # Calculate momentum direction entropy
        momentum_dir = np.sign(momentum)
        momentum_bins = pd.cut(momentum_dir, bins=[-2, -0.5, 0.5, 2], labels=False)
        momentum_prob = momentum_bins.rolling(window=window).apply(
            lambda x: -np.sum(pd.Series(x).value_counts(normalize=True) * 
                            np.log(pd.Series(x).value_counts(normalize=True) + 1e-10))
        )
        momentum_entropy_components.append(momentum_prob)
    
    momentum_entropy_combined = pd.concat(momentum_entropy_components, axis=1).mean(axis=1)
    
    # Momentum adaptation during regime shifts
    regime_momentum = momentum_entropy_combined * (1 - regime_breaks)
    
    # Volume entropy confirmation
    volume_confirmation = volume_entropy_combined.rolling(window=5).corr(vol_entropy_combined)
    
    # Final signal construction
    # Weight momentum by entropy regime stability and complexity
    regime_stability = 1 / (1 + regime_breaks)
    signal = regime_momentum * regime_stability * volume_confirmation
    
    # Normalize the final signal
    signal_normalized = (signal - signal.rolling(window=20).mean()) / signal.rolling(window=20).std()
    
    return signal_normalized
