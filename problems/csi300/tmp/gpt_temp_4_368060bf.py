import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Signature with Market Microstructure Entropy factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Frequency Volatility Decomposition
    data['ultra_short_vol'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['intraday_vol'] = (data['high'] - data['low']) / data['open']
    data['interday_vol'] = np.abs(data['close'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Handle division by zero
    data['interday_vol'] = data['interday_vol'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-Volatility Asymmetry
    data['volume_vol_ratio'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['vol_volume_corr'] = data['high'] - data['low'] / data['volume'].replace(0, np.nan)
    
    # Rolling correlation between volume and volatility (5-day window)
    data['vol_volume_rolling_corr'] = data['volume'].rolling(window=5).corr(data['high'] - data['low'])
    
    # Price Path Complexity
    data['wiggling_intensity'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['directional_persistence'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['close'].shift(1) - data['close'].shift(2))
    
    # Path efficiency ratio (3-day window)
    data['path_efficiency'] = np.abs(data['close'] - data['close'].shift(3)) / (
        np.abs(data['close'] - data['close'].shift(1)) + 
        np.abs(data['close'].shift(1) - data['close'].shift(2)) + 
        np.abs(data['close'].shift(2) - data['close'].shift(3))
    ).replace(0, np.nan)
    
    # Microstructure Entropy Measures
    # Volume-time distribution entropy approximation
    rolling_volume = data['volume'].rolling(window=10)
    data['volume_entropy'] = -rolling_volume.apply(
        lambda x: np.sum((x / x.sum()) * np.log((x / x.sum()).replace(0, 1))) if x.sum() > 0 else 0
    )
    
    # Price-level clustering entropy approximation
    price_ranges = (data['high'] - data['low']).rolling(window=5)
    data['price_clustering_entropy'] = -price_ranges.apply(
        lambda x: np.sum((x / x.sum()) * np.log((x / x.sum()).replace(0, 1))) if x.sum() > 0 else 0
    )
    
    # Bid-ask bounce frequency estimation (using open-close differences)
    data['bounce_frequency'] = (np.sign(data['close'] - data['open']) != np.sign(data['close'].shift(1) - data['open'].shift(1))).rolling(window=5).mean()
    
    # Order flow imbalance persistence (using volume-price relationship)
    data['ofi_persistence'] = (data['close'] * data['volume']).pct_change().rolling(window=5).std()
    
    # Regime-Dependent Signal Construction
    # Volatility regime detection
    vol_regime_threshold = data['ultra_short_vol'].rolling(window=20).quantile(0.7)
    data['high_vol_regime'] = (data['ultra_short_vol'] > vol_regime_threshold).astype(int)
    
    # Volume regime classification
    volume_regime_threshold = data['volume'].rolling(window=20).quantile(0.7)
    data['high_volume_regime'] = (data['volume'] > volume_regime_threshold).astype(int)
    
    # Entropy-based market state identification
    entropy_threshold = data['volume_entropy'].rolling(window=20).quantile(0.7)
    data['high_entropy_state'] = (data['volume_entropy'] > entropy_threshold).astype(int)
    
    # Alpha Synthesis
    # Volatility signature score
    volatility_score = (
        data['ultra_short_vol'].rank(pct=True) + 
        data['intraday_vol'].rank(pct=True) + 
        data['interday_vol'].rank(pct=True)
    ) / 3
    
    # Entropy-adjusted momentum
    momentum = data['close'].pct_change(periods=3)
    entropy_adjusted_momentum = momentum * (1 - data['volume_entropy'].rank(pct=True))
    
    # Regime-conditional signal enhancement
    regime_multiplier = 1 + 0.5 * data['high_vol_regime'] - 0.3 * data['high_volume_regime'] + 0.2 * data['high_entropy_state']
    
    # Multi-timeframe convergence detection
    short_trend = data['close'].pct_change(periods=3)
    medium_trend = data['close'].pct_change(periods=5)
    convergence_score = (np.sign(short_trend) == np.sign(medium_trend)).astype(float)
    
    # Final alpha factor construction
    alpha_factor = (
        0.4 * volatility_score +
        0.3 * entropy_adjusted_momentum.rank(pct=True) +
        0.2 * regime_multiplier +
        0.1 * convergence_score
    )
    
    # Normalize and handle missing values
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20).mean()) / alpha_factor.rolling(window=20).std()
    
    return alpha_factor
