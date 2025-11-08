import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns for volatility estimation
    returns = df['close'].pct_change()
    
    # Volatility Clustering Detection
    # 5-day rolling volatility (standard deviation of returns)
    vol_5d = returns.rolling(window=5).std()
    # Autocorrelation of volatility with lag 1
    vol_autocorr = vol_5d.rolling(window=10).apply(
        lambda x: x.autocorr(lag=1) if len(x.dropna()) >= 6 else np.nan, 
        raw=False
    )
    
    # Momentum Decay Analysis
    # 5-day and 10-day rate of change
    roc_5d = (df['close'] / df['close'].shift(5) - 1)
    roc_10d = (df['close'] / df['close'].shift(10) - 1)
    momentum_decay_rate = roc_5d / (roc_10d + 0.001)  # Avoid division by zero
    
    # Price efficiency: (close - open) / (high - low + 0.001)
    price_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
    
    # Volume acceleration: 5-day volume change rate
    volume_5d_roc = (df['volume'] / df['volume'].shift(5) - 1)
    
    # Cluster-Adaptive Signal Construction
    strong_clustering = vol_autocorr > 0.3
    weak_clustering = vol_autocorr < 0.1
    
    # Initialize signals
    reversal_signal = momentum_decay_rate * price_efficiency
    continuation_signal = momentum_decay_rate * volume_5d_roc
    
    # Combine signals based on clustering regime
    combined_signal = pd.Series(index=df.index, dtype=float)
    combined_signal[strong_clustering] = reversal_signal[strong_clustering]
    combined_signal[weak_clustering] = continuation_signal[weak_clustering]
    
    # For moderate clustering (0.1 <= persistence <= 0.3), use weighted average
    moderate_clustering = ~strong_clustering & ~weak_clustering
    weight = (vol_autocorr[moderate_clustering] - 0.1) / 0.2  # Normalize to [0,1]
    combined_signal[moderate_clustering] = (
        weight * reversal_signal[moderate_clustering] + 
        (1 - weight) * continuation_signal[moderate_clustering]
    )
    
    # Dynamic Weighting
    # Cluster strength weighting (absolute value of autocorrelation)
    cluster_weight = vol_autocorr.abs()
    
    # Amount-based decay adjustment (normalized amount)
    amount_normalized = df['amount'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 0.001), 
        raw=False
    )
    amount_weight = 1 / (1 + np.exp(-amount_normalized))  # Sigmoid normalization
    
    # Final factor: weighted signal
    final_factor = combined_signal * cluster_weight * amount_weight
    
    return final_factor
