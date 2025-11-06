import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Entropy Asymmetric Elasticity Factors
    """
    data = df.copy()
    
    # Calculate price entropy (rolling entropy of price changes)
    price_change = data['close'].pct_change().fillna(0)
    price_entropy = -price_change.rolling(window=5).apply(
        lambda x: np.sum(x * np.log(np.abs(x) + 1e-8)) if np.sum(np.abs(x)) > 0 else 0, raw=False
    ).fillna(0)
    
    # Calculate volume entropy (rolling entropy of volume changes)
    volume_change = data['volume'].pct_change().fillna(0)
    volume_entropy = -volume_change.rolling(window=5).apply(
        lambda x: np.sum(x * np.log(np.abs(x) + 1e-8)) if np.sum(np.abs(x)) > 0 else 0, raw=False
    ).fillna(0)
    
    # Volatility-Entropy Regime
    # True Range Entropy
    true_range = (data['high'] - data['low']) / data['close'].shift(1).fillna(method='bfill')
    true_range_entropy = true_range * price_entropy
    
    # Gap Volatility Entropy
    gap_volatility = np.abs(data['open'] - data['close'].shift(1).fillna(method='bfill')) / data['close'].shift(1).fillna(method='bfill')
    gap_volatility_entropy = gap_volatility * price_entropy
    
    # Volume-Range Entropy
    volume_range_entropy = ((data['high'] - data['low']) * data['volume'] / (data['amount'] + 1e-8)) * volume_entropy
    
    # Volume-Gap Entropy
    volume_gap_entropy = (np.abs(data['open'] - data['close'].shift(1).fillna(method='bfill')) * data['volume'] / (data['amount'] + 1e-8)) * volume_entropy
    
    # Asymmetric Elasticity
    # Directional Entropy Asymmetry
    up_entropy_volume = ((data['close'] > data['open']).astype(float) * data['volume'] / (data['amount'] + 1e-8)) * price_entropy
    down_entropy_volume = ((data['close'] < data['open']).astype(float) * data['volume'] / (data['amount'] + 1e-8)) * price_entropy
    
    # Elasticity Momentum
    entropy_asymmetry_change = up_entropy_volume - down_entropy_volume
    entropy_asymmetry_acceleration = entropy_asymmetry_change.diff().fillna(0)
    
    # Microstructure Entropy Imbalance
    # Price-Entropy Imbalance
    high_low_entropy_ratio = ((data['high'] - data['close']) / (data['close'] - data['low'] + 1e-8)) * price_entropy
    intraday_entropy_pressure = ((data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'] + 1e-8)) * price_entropy
    
    # Volume-Entropy Imbalance
    high_entropy_pressure = ((data['high'] - data['close']) * data['volume'] / (data['amount'] + 1e-8)) * volume_entropy
    low_entropy_support = ((data['close'] - data['low']) * data['volume'] / (data['amount'] + 1e-8)) * volume_entropy
    
    # Efficiency-Entropy Elasticity
    price_efficiency_entropy = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * price_entropy
    volume_efficiency_entropy = (data['volume'] / (data['amount'] / (data['high'] - data['low'] + 1e-8) + 1e-8)) * volume_entropy
    
    # Adaptive Factor Construction
    # Primary Factor
    high_vol_entropy = entropy_asymmetry_change * true_range_entropy
    low_vol_entropy = entropy_asymmetry_change * volume_range_entropy
    transition_elasticity = entropy_asymmetry_acceleration * np.abs(true_range_entropy - volume_range_entropy)
    
    # Secondary Factor
    high_efficiency = high_low_entropy_ratio * price_efficiency_entropy
    low_efficiency = intraday_entropy_pressure * volume_efficiency_entropy
    transition_imbalance = (high_entropy_pressure - low_entropy_support) * np.abs(price_efficiency_entropy - volume_efficiency_entropy)
    
    # Final Alpha - Regime-Weighted Elasticity Combination
    vol_entropy_regime = (true_range_entropy.rolling(window=10).mean() > volume_range_entropy.rolling(window=10).mean()).astype(float)
    
    primary_factor = vol_entropy_regime * high_vol_entropy + (1 - vol_entropy_regime) * low_vol_entropy + transition_elasticity
    secondary_factor = vol_entropy_regime * high_efficiency + (1 - vol_entropy_regime) * low_efficiency + transition_imbalance
    
    # Final alpha with entropy-volatility validation and asymmetric elasticity persistence
    final_alpha = (primary_factor.rolling(window=5).mean() * 
                  secondary_factor.rolling(window=5).mean() * 
                  entropy_asymmetry_change.rolling(window=3).mean())
    
    # Normalize and clean
    final_alpha = (final_alpha - final_alpha.rolling(window=20).mean()) / (final_alpha.rolling(window=20).std() + 1e-8)
    final_alpha = final_alpha.fillna(0)
    
    return final_alpha
