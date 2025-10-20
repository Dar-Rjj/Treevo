import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Acceleration with Volume-Amount Confirmation
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Momentum Acceleration
    # Exponential decay weighted returns
    def exp_weighted_return(window, decay):
        weights = np.array([decay ** i for i in range(window)][::-1])
        returns = data['close'].pct_change().rolling(window).apply(
            lambda x: np.sum(x * weights) if len(x) == window else np.nan
        )
        return returns
    
    # Calculate multi-scale momentum
    mom_5d = exp_weighted_return(5, 0.9)
    mom_20d = exp_weighted_return(20, 0.94)
    mom_60d = exp_weighted_return(60, 0.98)
    
    # Momentum patterns
    short_medium_divergence = mom_5d - mom_20d
    roc_10d = data['close'].pct_change(10)
    momentum_acceleration = roc_10d.pct_change(5)
    
    # Volume-Amount Coherence
    # Volume momentum analysis
    volume_5d_avg = data['volume'].rolling(5).mean()
    volume_20d_avg = data['volume'].rolling(20).mean()
    volume_ratio = volume_5d_avg / volume_20d_avg
    
    # Volume acceleration
    volume_ratio_prev = volume_ratio.shift(5)
    volume_acceleration = (volume_ratio - volume_ratio_prev) / volume_ratio_prev
    
    # Amount-based order flow
    amount_volume_ratio = data['amount'] / (data['volume'] + 1e-8)
    amount_clustering = amount_volume_ratio.rolling(10).std() / (amount_volume_ratio.rolling(20).mean() + 1e-8)
    
    # Volume-amount coherence score
    volume_amount_coherence = (volume_ratio * (1 + volume_acceleration) * 
                             (1 - np.abs(amount_clustering)))
    
    # Regime-Adaptive Signal Synthesis
    # Volatility regime classification
    volatility_20d = data['close'].pct_change().rolling(20).std()
    volatility_regime = (volatility_20d - volatility_20d.rolling(60).mean()) / volatility_20d.rolling(60).std()
    
    # Base momentum acceleration factor
    base_momentum = (short_medium_divergence * 0.4 + 
                    momentum_acceleration * 0.3 + 
                    (mom_5d - mom_60d) * 0.3)
    
    # Composite alpha generation with regime-adaptive weighting
    low_vol_regime = (volatility_regime < -0.5).astype(int)
    high_vol_regime = (volatility_regime > 0.5).astype(int)
    normal_regime = 1 - low_vol_regime - high_vol_regime
    
    # Regime-adaptive weights
    momentum_weight = (low_vol_regime * 0.7 + normal_regime * 0.5 + high_vol_regime * 0.3)
    volume_weight = (low_vol_regime * 0.3 + normal_regime * 0.5 + high_vol_regime * 0.7)
    
    # Final composite alpha
    composite_alpha = (base_momentum * momentum_weight + 
                      volume_amount_coherence * volume_weight)
    
    # Normalize the final factor
    alpha_series = (composite_alpha - composite_alpha.rolling(60).mean()) / composite_alpha.rolling(60).std()
    
    return alpha_series
