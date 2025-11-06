import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Volatility Regime Classification
    volatility_score = returns.rolling(window=20, min_periods=10).std()
    
    # Historical volatility percentiles (252-day lookback)
    vol_60_percentile = volatility_score.rolling(window=252, min_periods=50).quantile(0.6)
    vol_40_percentile = volatility_score.rolling(window=252, min_periods=50).quantile(0.4)
    
    # Regime classification
    high_vol_regime = volatility_score > vol_60_percentile
    low_vol_regime = volatility_score < vol_40_percentile
    normal_vol_regime = ~high_vol_regime & ~low_vol_regime
    
    # Price Path Entropy
    price_changes = df['close'].pct_change().rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(-p * np.log(p + 1e-10) if p > 0 else 0 
                        for p in (np.abs(x) / (np.sum(np.abs(x)) + 1e-10))), 
        raw=False
    )
    
    # Volume Distribution Entropy
    volume_entropy = df['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: np.sum(-p * np.log(p + 1e-10) if p > 0 else 0 
                        for p in (x / (np.sum(x) + 1e-10))), 
        raw=False
    )
    
    # Entropy Divergence (KL divergence approximation)
    def kl_divergence_approx(price_ent, vol_ent):
        if price_ent > 0 and vol_ent > 0:
            return price_ent * np.log((price_ent + 1e-10) / (vol_ent + 1e-10))
        return 0
    
    entropy_divergence = pd.Series([
        kl_divergence_approx(p, v) for p, v in zip(price_changes, volume_entropy)
    ], index=df.index)
    
    # Regime-Adaptive Signal Construction
    # 3-day momentum persistence for high volatility
    momentum_3d = df['close'].pct_change(3)
    
    # 5-day mean reversion strength for low volatility
    mean_reversion_5d = -df['close'].pct_change(5) / df['close'].pct_change(5).rolling(window=20, min_periods=10).std()
    
    # 2-day trend following for normal volatility
    trend_2d = df['close'].pct_change(2)
    
    # Combine signals based on regime
    raw_signal = pd.Series(0.0, index=df.index)
    raw_signal[high_vol_regime] = entropy_divergence[high_vol_regime] * momentum_3d[high_vol_regime]
    raw_signal[low_vol_regime] = price_changes[low_vol_regime] * mean_reversion_5d[low_vol_regime]
    raw_signal[normal_vol_regime] = volume_entropy[normal_vol_regime] * trend_2d[normal_vol_regime]
    
    # Liquidity-Adjusted Enhancement
    amihud_ratio = np.abs(returns) / (df['amount'] + 1e-10)
    liquidity_weighted_signal = raw_signal * (1 / (1 + amihud_ratio))
    
    # Regime Persistence Filter
    regime_duration = pd.Series(0, index=df.index)
    current_regime = None
    current_duration = 0
    
    for i in range(len(df)):
        if high_vol_regime.iloc[i]:
            regime = 'high'
        elif low_vol_regime.iloc[i]:
            regime = 'low'
        else:
            regime = 'normal'
        
        if regime == current_regime:
            current_duration += 1
        else:
            current_duration = 1
            current_regime = regime
        
        regime_duration.iloc[i] = current_duration
    
    # Duration adjustment
    duration_adjustment = np.minimum(1, regime_duration / 10)
    final_signal = liquidity_weighted_signal * duration_adjustment
    
    return final_signal
