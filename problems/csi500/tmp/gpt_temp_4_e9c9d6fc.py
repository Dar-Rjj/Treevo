import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Volume-Weighted Reversal factor
    """
    # Initialize output series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Small epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Calculate daily returns for volatility computation
    returns = df['close'].pct_change()
    
    # Volatility-Normalized Price Reversal
    # 5-day rolling volatility (using returns from t-6 to t-1)
    vol_5d = returns.shift(1).rolling(window=5).std()
    
    # 10-day rolling volatility (using returns from t-11 to t-1)
    vol_10d = returns.shift(1).rolling(window=10).std()
    
    # Short-term reversal (3-day)
    reversal_3d = (df['close'].shift(1) - df['close'].shift(3)) / (df['close'].shift(3) + epsilon)
    normalized_reversal_3d = reversal_3d / (vol_5d + epsilon)
    
    # Medium-term reversal (8-day)
    reversal_8d = (df['close'].shift(1) - df['close'].shift(8)) / (df['close'].shift(8) + epsilon)
    normalized_reversal_8d = reversal_8d / (vol_10d + epsilon)
    
    # Regime detection and reversal selection
    high_vol_regime = vol_5d > vol_10d
    selected_reversal = pd.Series(index=df.index, dtype=float)
    selected_reversal[high_vol_regime] = normalized_reversal_3d[high_vol_regime]
    selected_reversal[~high_vol_regime] = normalized_reversal_8d[~high_vol_regime]
    
    # Volume-Price Divergence
    # Volume acceleration signal
    volume_change_3d = df['volume'] / (df['volume'].shift(3) + epsilon) - 1
    volume_change_8d = df['volume'] / (df['volume'].shift(8) + epsilon) - 1
    volume_divergence = volume_change_3d - volume_change_8d
    
    # Price-volume relationship (5-day rolling correlation)
    price_returns = df['close'].pct_change()
    volume_returns = df['volume'].pct_change()
    
    # Calculate rolling correlation
    correlation = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        start_idx = i - 4
        end_idx = i
        price_window = price_returns.iloc[start_idx:end_idx+1]
        volume_window = volume_returns.iloc[start_idx:end_idx+1]
        if len(price_window) >= 3 and len(volume_window) >= 3:
            corr_val = price_window.corr(volume_window)
            correlation.iloc[i] = corr_val if not np.isnan(corr_val) else 0
    
    # Correlation-weighted divergence
    correlation_weighted_divergence = volume_divergence * np.abs(correlation)
    
    # Nonlinear Volume Weighting
    # 10-day volume percentile
    volume_percentile = df['volume'].rolling(window=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1]), raw=False
    )
    
    # Adaptive weighting function
    adaptive_weight = pd.Series(index=df.index, dtype=float)
    high_volume = volume_percentile > 0.7
    medium_volume = (volume_percentile >= 0.3) & (volume_percentile <= 0.7)
    low_volume = volume_percentile < 0.3
    
    adaptive_weight[high_volume] = np.power(volume_divergence[high_volume], 2)
    adaptive_weight[medium_volume] = volume_divergence[medium_volume]
    adaptive_weight[low_volume] = np.power(np.abs(volume_divergence[low_volume]), 0.5) * np.sign(volume_divergence[low_volume])
    
    # Final Alpha Construction
    # Base factor
    base_factor = selected_reversal * correlation_weighted_divergence
    
    # Volume weighted factor
    weighted_factor = base_factor * adaptive_weight
    
    # Regime-based sign adjustment
    positive_correlation = correlation > 0
    final_factor = pd.Series(index=df.index, dtype=float)
    final_factor[positive_correlation] = weighted_factor[positive_correlation]
    final_factor[~positive_correlation] = -weighted_factor[~positive_correlation]
    
    return final_factor
