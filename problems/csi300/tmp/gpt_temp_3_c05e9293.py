import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Volatility-Volume Asymmetry Factor
    Multi-frequency analysis of volatility patterns and asymmetric volume dynamics
    """
    data = df.copy()
    
    # Calculate returns and volume features
    data['returns'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    
    # Multi-Frequency Volatility Analysis
    # High-frequency volatility (1-5 days)
    data['vol_hf'] = data['returns'].rolling(window=5, min_periods=3).std()
    
    # Medium-frequency volatility (5-20 days)
    data['vol_mf'] = data['returns'].rolling(window=20, min_periods=10).std()
    
    # Low-frequency volatility (20-60 days)
    data['vol_lf'] = data['returns'].rolling(window=60, min_periods=30).std()
    
    # Volatility persistence patterns
    data['vol_clustering'] = data['vol_hf'].rolling(window=10).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
    )
    
    # Mean-reversion strength
    data['vol_mean_reversion'] = (data['vol_hf'] - data['vol_hf'].rolling(window=20).mean()) / data['vol_hf'].rolling(window=20).std()
    
    # Asymmetric Volume Dynamics
    # Directional volume analysis
    data['up_day_volume'] = data['volume'].where(data['returns'] > 0, 0)
    data['down_day_volume'] = data['volume'].where(data['returns'] < 0, 0)
    
    data['up_volume_concentration'] = data['up_day_volume'].rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
    data['down_volume_intensity'] = data['down_day_volume'].rolling(window=20).mean() / data['volume'].rolling(window=20).mean()
    
    # Volume distribution skewness
    data['volume_skew'] = data['volume'].rolling(window=30).apply(
        lambda x: pd.Series(x).skew() if len(x) > 10 else 0
    )
    
    # Volatility-Volume Coupling
    # Volatility-driven volume response
    high_vol_threshold = data['vol_hf'].rolling(window=60).quantile(0.7)
    low_vol_threshold = data['vol_hf'].rolling(window=60).quantile(0.3)
    
    data['high_vol_volume_response'] = data['volume_change'].where(data['vol_hf'] > high_vol_threshold, 0)
    data['low_vol_volume_behavior'] = data['volume_change'].where(data['vol_hf'] < low_vol_threshold, 0)
    
    # Volume-induced volatility
    heavy_volume_threshold = data['volume'].rolling(window=60).quantile(0.7)
    light_volume_threshold = data['volume'].rolling(window=60).quantile(0.3)
    
    data['heavy_vol_impact'] = data['vol_hf'].where(data['volume'] > heavy_volume_threshold, 0)
    data['light_vol_suppression'] = data['vol_hf'].where(data['volume'] < light_volume_threshold, 0)
    
    # Cross-Frequency Interactions
    data['volatility_spread'] = (data['vol_hf'] - data['vol_lf']) / data['vol_lf']
    
    # Volume-volatility phase relationship
    data['volume_vol_correlation'] = data['volume'].rolling(window=20).corr(data['vol_hf'])
    
    # Frequency-dependent asymmetry
    data['hf_vol_asymmetry'] = (data['vol_hf'].where(data['returns'] > 0, 0) - 
                               data['vol_hf'].where(data['returns'] < 0, 0)).rolling(window=10).mean()
    
    # Regime-Adaptive Framework
    # Volatility regime identification
    vol_regime_threshold = data['vol_hf'].rolling(window=60).quantile(0.6)
    data['high_vol_regime'] = (data['vol_hf'] > vol_regime_threshold).astype(int)
    
    # Volume regime classification
    volume_regime_threshold = data['volume'].rolling(window=60).quantile(0.6)
    data['high_volume_regime'] = (data['volume'] > volume_regime_threshold).astype(int)
    
    # Regime transition signals
    data['regime_transition'] = (data['high_vol_regime'].diff() != 0) | (data['high_volume_regime'].diff() != 0)
    
    # Asymmetry Amplification
    # Volatility asymmetry magnification
    data['vol_asymmetry_magnified'] = data['hf_vol_asymmetry'] * data['vol_clustering']
    
    # Volume asymmetry enhancement
    data['volume_asymmetry_enhanced'] = (data['up_volume_concentration'] - data['down_volume_intensity']) * data['volume_skew']
    
    # Coupled asymmetry effects
    data['coupled_asymmetry'] = data['vol_asymmetry_magnified'] * data['volume_asymmetry_enhanced']
    
    # Signal Synthesis
    # Multi-dimensional asymmetry scoring
    volatility_asymmetry_score = (
        data['vol_asymmetry_magnified'].fillna(0) + 
        data['volatility_spread'].fillna(0) + 
        data['hf_vol_asymmetry'].fillna(0)
    ) / 3
    
    volume_asymmetry_score = (
        data['volume_asymmetry_enhanced'].fillna(0) + 
        data['up_volume_concentration'].fillna(0) - 
        data['down_volume_intensity'].fillna(0)
    ) / 3
    
    coupling_asymmetry_score = (
        data['coupled_asymmetry'].fillna(0) + 
        data['volume_vol_correlation'].fillna(0) + 
        data['high_vol_volume_response'].fillna(0) - 
        data['low_vol_volume_behavior'].fillna(0)
    ) / 4
    
    # Final factor synthesis with regime adaptation
    regime_weight = 1 + 0.5 * (data['high_vol_regime'] + data['high_volume_regime'])
    
    final_factor = (
        volatility_asymmetry_score * 0.4 +
        volume_asymmetry_score * 0.35 +
        coupling_asymmetry_score * 0.25
    ) * regime_weight
    
    # Apply regime transition smoothing
    final_factor = final_factor.where(~data['regime_transition'], final_factor.rolling(window=5).mean())
    
    # Normalize the final factor
    final_factor_normalized = (final_factor - final_factor.rolling(window=60).mean()) / final_factor.rolling(window=60).std()
    
    return final_factor_normalized
