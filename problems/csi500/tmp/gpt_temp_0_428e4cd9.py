import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Cross-Association Fractor
    Combines volatility-regime analysis, fractal divergence, and efficiency metrics
    to generate predictive alpha signals.
    """
    data = df.copy()
    
    # Volatility regime calculation (20-day rolling standard deviation)
    volatility = data['close'].rolling(window=20).std()
    high_vol_threshold = volatility.rolling(window=50).quantile(0.7)
    low_vol_threshold = volatility.rolling(window=50).quantile(0.3)
    
    # Volatility regime mask
    high_vol_regime = volatility > high_vol_threshold
    low_vol_regime = volatility < low_vol_threshold
    
    # 1. Volatility-Regime Price-Volume Dislocation Analysis
    # Regime-specific directional divergence
    # Price momentum
    price_momentum = pd.Series(index=data.index, dtype=float)
    price_momentum[high_vol_regime] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    price_momentum[low_vol_regime] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    price_momentum.fillna(0, inplace=True)
    
    # Volume momentum
    volume_momentum = pd.Series(index=data.index, dtype=float)
    volume_momentum[high_vol_regime] = (data['volume'] - data['volume'].shift(3)) / (data['volume'].shift(3) + 1e-8)
    volume_momentum[low_vol_regime] = (data['volume'] - data['volume'].shift(10)) / (data['volume'].shift(10) + 1e-8)
    volume_momentum.fillna(0, inplace=True)
    
    # Fractal-Enhanced Divergence Assessment
    # Price Fractal Dimension (FD)
    def calculate_fractal_dimension(series, window=5, range_window=3):
        path_length = series.diff().abs().rolling(window=window).sum()
        range_sum = (series.rolling(window=range_window).max() - 
                    series.rolling(window=range_window).min()).rolling(window=window).sum()
        fd = 1 + np.log(path_length + 1e-8) / np.log(range_sum + 1e-8)
        return fd
    
    price_fd = calculate_fractal_dimension(data['close'])
    volume_fd = calculate_fractal_dimension(data['volume'])
    
    # FD Momentum Divergence
    price_fd_momentum_div = (price_fd - price_fd.shift(3)) - (price_fd - price_fd.shift(5))
    volume_fd_momentum_div = (volume_fd - volume_fd.shift(3)) - (volume_fd - volume_fd.shift(5))
    
    # Multi-Dimensional Divergence Integration
    # Price-Volume Direction Mismatch Score
    direction_mismatch = ((price_momentum * volume_momentum < 0).astype(float) * 
                         (np.abs(price_fd_momentum_div) + np.abs(volume_fd_momentum_div)))
    
    # Fractal-Persistence Consistency Check
    fractal_consistency = ((np.sign(price_fd_momentum_div) == np.sign(price_momentum)).astype(float) + 
                          (np.sign(volume_fd_momentum_div) == np.sign(volume_momentum)).astype(float)) / 2
    
    divergence_confidence = direction_mismatch * fractal_consistency
    
    # 2. Fractal Range Efficiency Analysis
    # Price Movement Efficiency
    basic_efficiency = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Fractal-Adjusted Efficiency
    fractal_efficiency = basic_efficiency * price_fd
    
    # Volume Distribution Analysis
    volume_per_price = data['amount'] / (data['close'] + 1e-8)
    volume_concentration = volume_per_price.rolling(window=5).std() / (volume_per_price.rolling(window=5).mean() + 1e-8)
    
    # Efficiency-Fractal Correlation
    efficiency_fractal_corr = basic_efficiency.rolling(window=10).corr(price_fd)
    
    # Regime-Adaptive Efficiency Interpretation
    efficiency_score = pd.Series(index=data.index, dtype=float)
    efficiency_score[high_vol_regime] = (fractal_efficiency * 
                                        (1 + volume_fd) * 
                                        (1 + np.abs(efficiency_fractal_corr)))
    efficiency_score[low_vol_regime] = (basic_efficiency * 
                                       price_fd * 
                                       (1 - np.abs(efficiency_fractal_corr)))
    efficiency_score.fillna(0, inplace=True)
    
    # 3. Cross-Association Fractal Momentum
    # Fractal Momentum Persistence
    fd_direction_consistency = (np.sign(price_fd.diff()) == 
                               np.sign(price_fd.diff().shift(1))).rolling(window=3).mean()
    fd_persistence_ratio = fd_direction_consistency
    
    # Volume Fractal Support
    vfd_alignment = (np.sign(volume_fd.diff()) == np.sign(price_fd.diff())).astype(float)
    fractal_support_score = (fd_persistence_ratio + vfd_alignment) / 2
    
    # Regime-Adaptive Momentum Quality
    fd_variance = price_fd.rolling(window=5).std()
    fd_trend_consistency = price_fd.rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0
    )
    
    momentum_quality = pd.Series(index=data.index, dtype=float)
    momentum_quality[high_vol_regime] = 1 / (1 + fd_variance)
    momentum_quality[low_vol_regime] = np.abs(fd_trend_consistency)
    momentum_quality.fillna(0, inplace=True)
    
    # Cross-Dimensional Association
    fractal_divergence_corr = price_fd_momentum_div.rolling(window=10).corr(divergence_confidence)
    efficiency_fractal_integration = efficiency_score * fractal_support_score
    
    # 4. Regime-Adaptive Cross-Factor Synthesis
    # Multi-Dimensional Factor Combination
    # Fractal-Divergence Adjusted Momentum
    momentum_adjusted = price_momentum * (1 + divergence_confidence * np.sign(price_momentum))
    
    # Efficiency-Weighted Fractal Persistence
    efficiency_weighted_persistence = fd_persistence_ratio * efficiency_score
    
    # Volume Fractal Confirmation
    volume_confirmation = volume_fd * fractal_support_score
    
    # Volatility-Regime Final Integration
    final_factor = pd.Series(index=data.index, dtype=float)
    
    # High Volatility Factor Enhancement
    high_vol_factor = (momentum_adjusted * 
                      (1 + volume_confirmation) * 
                      (1 + divergence_confidence) * 
                      fd_persistence_ratio)
    
    # Low Volatility Factor Refinement
    low_vol_factor = (momentum_adjusted * 
                     price_fd * 
                     efficiency_weighted_persistence * 
                     np.abs(fd_trend_consistency))
    
    # Apply regime-specific factors
    final_factor[high_vol_regime] = high_vol_factor[high_vol_regime]
    final_factor[low_vol_regime] = low_vol_factor[low_vol_regime]
    final_factor.fillna(0, inplace=True)
    
    # Normalize the final factor
    final_factor = (final_factor - final_factor.rolling(window=50).mean()) / (final_factor.rolling(window=50).std() + 1e-8)
    
    return final_factor
