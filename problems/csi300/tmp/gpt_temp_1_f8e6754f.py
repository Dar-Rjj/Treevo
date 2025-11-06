import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import entropy

def heuristics_v2(df):
    """
    Volatility-Constrained Microstructure Entropy with Regime-Dependent Liquidity Spillover
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility-Constrained Microstructure Entropy
    # Price-Level Entropy Measurement
    # High-Low Range Entropy Concentration
    daily_range = (data['high'] - data['low']) / data['close']
    range_entropy = daily_range.rolling(window=5).apply(
        lambda x: entropy(np.histogram(x.dropna(), bins=5)[0] + 1e-10)
    )
    
    # Close-to-Close Return Dispersion Entropy
    returns = data['close'].pct_change()
    return_volatility = returns.rolling(window=10).std()
    return_entropy = returns.rolling(window=5).apply(
        lambda x: entropy(np.histogram(x.dropna(), bins=5)[0] + 1e-10)
    )
    
    # Intraday Price Path Fractality
    intraday_range = (data['high'] - data['low']) / (data['open'] + 1e-10)
    open_close_range = abs(data['close'] - data['open']) / data['open']
    path_complexity = intraday_range / (open_close_range + 1e-10)
    fractality_entropy = path_complexity.rolling(window=5).apply(
        lambda x: entropy(np.histogram(x.dropna(), bins=5)[0] + 1e-10)
    )
    
    # Volume-Flow Entropy Dynamics
    # Volume Distribution Asymmetry Entropy
    volume_ma = data['volume'].rolling(window=10).mean()
    volume_std = data['volume'].rolling(window=10).std()
    volume_zscore = (data['volume'] - volume_ma) / (volume_std + 1e-10)
    volume_entropy = volume_zscore.rolling(window=5).apply(
        lambda x: entropy(np.histogram(x.dropna(), bins=5)[0] + 1e-10)
    )
    
    # Order Flow Imbalance Persistence
    amount_per_volume = data['amount'] / (data['volume'] + 1e-10)
    order_flow_imbalance = amount_per_volume.rolling(window=5).std()
    order_flow_entropy = order_flow_imbalance.rolling(window=5).apply(
        lambda x: entropy(np.histogram(x.dropna(), bins=5)[0] + 1e-10)
    )
    
    # Volatility-Entropy Coupling
    volatility_regime = return_volatility.rolling(window=10).apply(
        lambda x: 1 if x.mean() > x.median() else 0
    )
    
    # High Volatility Entropy Compression
    high_vol_compression = (range_entropy * (1 - volatility_regime)).fillna(0)
    
    # Low Volatility Entropy Expansion  
    low_vol_expansion = (return_entropy * volatility_regime).fillna(0)
    
    # Volatility-Regime Entropy Transition
    vol_regime_change = volatility_regime.diff().abs().rolling(window=5).sum()
    
    # 2. Regime-Dependent Liquidity Spillover
    # Liquidity proxies
    dollar_volume = data['amount']
    volume_turnover = data['volume'] / data['volume'].rolling(window=20).mean()
    
    # Liquidity regime detection
    liquidity_ma = dollar_volume.rolling(window=10).mean()
    liquidity_std = dollar_volume.rolling(window=10).std()
    liquidity_regime = (dollar_volume - liquidity_ma) / (liquidity_std + 1e-10)
    
    # Time-Varying Liquidity Regimes
    # High-Frequency Liquidity Fragmentation
    intraday_liquidity = (data['high'] - data['low']) / dollar_volume
    liquidity_fragmentation = intraday_liquidity.rolling(window=5).std()
    
    # End-of-Day Liquidity Compression
    close_liquidity = abs(data['close'] - data['open']) / dollar_volume
    eod_compression = close_liquidity.rolling(window=5).mean()
    
    # Liquidity-Momentum Feedback Loops
    price_momentum = data['close'].pct_change(periods=5)
    liquidity_momentum = dollar_volume.pct_change(periods=5)
    liquidity_feedback = price_momentum * liquidity_momentum
    
    # 3. Microstructure-Liquidity Integration Framework
    # Entropy-Liquidity Regime Mapping
    high_entropy_high_liquidity = (range_entropy * liquidity_regime).fillna(0)
    low_entropy_constrained = (return_entropy * (1 - abs(liquidity_regime))).fillna(0)
    
    # Entropy-Liquidity Divergence Detection
    entropy_liquidity_corr = range_entropy.rolling(window=10).corr(liquidity_regime)
    
    # Volatility-Adjusted Signal Weighting
    vol_adjusted_weight = 1 / (return_volatility + 1e-10)
    
    # Multi-Timeframe Signal Synchronization
    # Intraday co-movement
    intraday_correlation = (range_entropy.rolling(window=5).corr(liquidity_fragmentation)).fillna(0)
    
    # 4. Composite Alpha Construction
    # Entropy-Liquidity Interaction Terms
    interaction_1 = range_entropy * liquidity_regime * vol_adjusted_weight
    interaction_2 = return_entropy * vol_regime_change * liquidity_feedback
    interaction_3 = fractality_entropy * intraday_correlation
    
    # Multi-Regime Signal Combination
    high_vol_high_entropy = interaction_1 * volatility_regime
    low_vol_constrained = interaction_2 * (1 - volatility_regime)
    transition_signals = interaction_3 * vol_regime_change
    
    # Dynamic Factor Allocation
    # Regime-adaptive weighting
    regime_weight = abs(liquidity_regime) / (abs(liquidity_regime).rolling(window=20).mean() + 1e-10)
    
    # Volatility-dependent weighting
    vol_weight = 1 / (return_volatility.rolling(window=10).mean() + 1e-10)
    
    # Combine all components
    composite_alpha = (
        high_vol_high_entropy * regime_weight * 0.3 +
        low_vol_constrained * vol_weight * 0.3 +
        transition_signals * regime_weight * vol_weight * 0.4 +
        entropy_liquidity_corr * 0.1 +
        high_entropy_high_liquidity * 0.1 -
        low_entropy_constrained * 0.1
    )
    
    # Normalize and clean
    alpha_series = composite_alpha.fillna(0)
    alpha_series = (alpha_series - alpha_series.rolling(window=20).mean()) / (alpha_series.rolling(window=20).std() + 1e-10)
    
    return alpha_series
