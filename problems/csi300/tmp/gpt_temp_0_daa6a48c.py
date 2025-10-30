import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining volatility-adjusted momentum, 
    price-volume efficiency divergence, and dynamic regime-based weighting.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility-Normalized Momentum
    # Daily returns for volatility calculation
    daily_returns = data['close'].pct_change()
    
    # 5-day momentum components
    mom_5_raw = data['close'] / data['close'].shift(5) - 1
    vol_5 = daily_returns.rolling(window=5).std()
    mom_5_norm = mom_5_raw / (vol_5 + 1e-8)
    
    # 20-day momentum components
    mom_20_raw = data['close'] / data['close'].shift(20) - 1
    vol_20 = daily_returns.rolling(window=20).std()
    mom_20_norm = mom_20_raw / (vol_20 + 1e-8)
    
    # Momentum convergence (20-day correlation window)
    mom_convergence = pd.Series(index=data.index, dtype=float)
    for i in range(20, len(data)):
        start_idx = max(0, i-19)
        mom_5_window = mom_5_norm.iloc[start_idx:i+1]
        mom_20_window = mom_20_norm.iloc[start_idx:i+1]
        if len(mom_5_window) > 5 and len(mom_20_window) > 5:
            corr = mom_5_window.corr(mom_20_window)
            avg_mom = (mom_5_norm.iloc[i] + mom_20_norm.iloc[i]) / 2
            mom_convergence.iloc[i] = corr * avg_mom if not np.isnan(corr) else 0
    
    # Volatility regime detection
    vol_60 = daily_returns.rolling(window=60).std()
    high_vol_regime = vol_20 > (1.5 * vol_60)
    low_vol_regime = vol_20 < (0.67 * vol_60)
    normal_vol_regime = ~high_vol_regime & ~low_vol_regime
    
    # Regime-weighted momentum
    regime_momentum = pd.Series(index=data.index, dtype=float)
    regime_momentum[high_vol_regime] = 0.7 * mom_5_norm + 0.3 * mom_20_norm
    regime_momentum[low_vol_regime] = 0.3 * mom_5_norm + 0.7 * mom_20_norm
    regime_momentum[normal_vol_regime] = 0.5 * mom_5_norm + 0.5 * mom_20_norm
    
    # 2. Price-Volume Efficiency Divergence
    # Efficiency metrics
    price_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    volume_efficiency = data['amount'] / (data['high'] - data['low'] + 1e-8)
    combined_efficiency = price_efficiency * volume_efficiency
    
    # Efficiency trends
    eff_mom_5 = combined_efficiency / combined_efficiency.shift(5) - 1
    eff_mom_20 = combined_efficiency / combined_efficiency.shift(20) - 1
    
    # Efficiency convergence
    eff_convergence = pd.Series(index=data.index, dtype=float)
    for i in range(20, len(data)):
        start_idx = max(0, i-19)
        eff_5_window = eff_mom_5.iloc[start_idx:i+1]
        eff_20_window = eff_mom_20.iloc[start_idx:i+1]
        if len(eff_5_window) > 5 and len(eff_20_window) > 5:
            corr = eff_5_window.corr(eff_20_window)
            eff_convergence.iloc[i] = corr if not np.isnan(corr) else 0
    
    # Price-volume divergence
    pv_divergence = pd.Series(index=data.index, dtype=float)
    for i in range(20, len(data)):
        start_idx = max(0, i-19)
        eff_window = price_efficiency.iloc[start_idx:i+1]
        vol_window = data['volume'].iloc[start_idx:i+1].pct_change()
        if len(eff_window) > 5 and len(vol_window) > 5:
            corr = eff_window.corr(vol_window)
            avg_momentum = (mom_5_norm.iloc[i] + mom_20_norm.iloc[i]) / 2
            avg_efficiency_mom = (eff_mom_5.iloc[i] + eff_mom_20.iloc[i]) / 2
            divergence = avg_momentum - avg_efficiency_mom
            pv_divergence.iloc[i] = divergence * corr if not np.isnan(corr) else divergence
    
    # 3. Dynamic Signal Weighting System
    # Rolling correlations for dynamic weighting
    mom_eff_corr = pd.Series(index=data.index, dtype=float)
    mom_range_corr = pd.Series(index=data.index, dtype=float)
    
    for i in range(20, len(data)):
        start_idx = max(0, i-19)
        mom_window = regime_momentum.iloc[start_idx:i+1]
        eff_window = combined_efficiency.iloc[start_idx:i+1]
        range_window = (data['high'] - data['low']).iloc[start_idx:i+1]
        
        if len(mom_window) > 5:
            mom_eff_corr.iloc[i] = mom_window.corr(eff_window) if len(mom_window) > 5 else 0
            mom_range_corr.iloc[i] = mom_window.corr(range_window) if len(mom_window) > 5 else 0
    
    # Recent signal efficiency (simplified)
    signal_efficiency = pd.Series(index=data.index, dtype=float)
    for i in range(25, len(data)):
        signal_window = regime_momentum.iloc[i-5:i]
        return_window = daily_returns.iloc[i-4:i+1]
        if len(signal_window) > 3 and len(return_window) > 3:
            corr = signal_window.corr(return_window)
            signal_efficiency.iloc[i] = abs(corr) if not np.isnan(corr) else 0
    
    # Dynamic weights based on regime and correlations
    momentum_weight = pd.Series(0.4, index=data.index)
    efficiency_weight = pd.Series(0.4, index=data.index)
    divergence_weight = pd.Series(0.2, index=data.index)
    
    # Adjust weights based on volatility regime
    momentum_weight[high_vol_regime] = 0.3
    efficiency_weight[high_vol_regime] = 0.3
    divergence_weight[high_vol_regime] = 0.4
    
    momentum_weight[low_vol_regime] = 0.5
    efficiency_weight[low_vol_regime] = 0.4
    divergence_weight[low_vol_regime] = 0.1
    
    # Regime confidence multiplier
    regime_confidence = pd.Series(1.0, index=data.index)
    regime_confidence[high_vol_regime | low_vol_regime] = 1.2
    regime_confidence[normal_vol_regime] = 1.0
    
    # 4. Composite Alpha Factor
    # Core signal components
    momentum_score = regime_momentum * (1 + 0.5 * mom_convergence.fillna(0))
    efficiency_score = combined_efficiency * (1 + 0.3 * eff_convergence.fillna(0))
    divergence_score = pv_divergence.fillna(0)
    
    # Apply dynamic weights
    weighted_momentum = momentum_score * momentum_weight
    weighted_efficiency = efficiency_score * efficiency_weight
    weighted_divergence = divergence_score * divergence_weight
    
    # Final alpha calculation
    alpha_factor = (weighted_momentum + weighted_efficiency + weighted_divergence) * regime_confidence
    
    # Cross-sectional normalization (z-score)
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20, min_periods=10).mean()) / (alpha_factor.rolling(window=20, min_periods=10).std() + 1e-8)
    
    return alpha_factor
