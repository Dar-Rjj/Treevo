import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Volatility-Microstructure Dynamics
    # Volatility-Efficiency Correlation Regimes
    for window in [5, 10]:
        volatility = data['high'].rolling(window=window).max() - data['low'].rolling(window=window).min()
        efficiency = (data['close'].rolling(window=window).mean() - data['open'].rolling(window=window).mean()) / volatility
        data[f'vol_eff_corr_{window}'] = volatility.rolling(window=window).corr(efficiency)
    
    data['vol_eff_regime_shift'] = data['vol_eff_corr_5'] - data['vol_eff_corr_10']
    
    # Volume-Weighted Volatility Momentum
    vol_weighted_vol_eff = ((data['high'] - data['low']) / data['close'].shift(5) * data['volume']).rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    data['vol_eff_momentum_alignment'] = np.sign(data['high'] - data['low']) * np.sign(vol_weighted_vol_eff)
    
    # Volume-Volatility Acceleration & Regime Persistence
    vol_vol_ratio = data['volume'] / (data['high'] - data['low'])
    data['vol_vol_momentum_3d'] = vol_vol_ratio / vol_vol_ratio.shift(3) - 1
    data['vol_vol_momentum_5d'] = vol_vol_ratio / vol_vol_ratio.shift(5) - 1
    
    # Volume-Volatility Regime Persistence
    vol_vol_increase = vol_vol_ratio > vol_vol_ratio.shift(1)
    data['vol_vol_regime_persistence'] = vol_vol_increase.rolling(window=10, min_periods=1).apply(lambda x: len(x) - np.argmin(x[::-1]) if np.any(x) else 0)
    
    # Intraday Volatility-Microstructure Efficiency
    data['gap_vol_size'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_vol_efficiency'] = abs(data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1)).replace(0, np.nan)
    
    daily_vol_range = data['high'] - data['low']
    close_open_dist = abs(data['close'] - data['open'])
    data['pure_vol_efficiency'] = close_open_dist / daily_vol_range.replace(0, np.nan)
    data['volume_scaled_vol_efficiency'] = data['pure_vol_efficiency'] * data['volume'] / daily_vol_range.replace(0, np.nan)
    
    vol_midpoint = (data['high'] + data['low']) / 2
    data['intraday_vol_position'] = (data['close'] - vol_midpoint) / daily_vol_range.replace(0, np.nan)
    data['price_spread_proxy'] = 2 * abs(data['close'] - vol_midpoint) / data['close']
    
    # Multi-Timeframe Price-Volatility Divergence
    vol_eff_5d = (data['high'] - data['low']) / data['close']
    vol_eff_10d = (data['high'].shift(5) - data['low'].shift(5)) / data['close'].shift(5)
    data['vol_eff_roc_5d'] = vol_eff_5d - vol_eff_10d
    
    vol_eff_15d = (data['high'].shift(10) - data['low'].shift(10)) / data['close'].shift(10)
    data['vol_eff_roc_10d'] = vol_eff_5d - vol_eff_15d
    data['vol_eff_momentum_divergence'] = data['vol_eff_roc_5d'] - data['vol_eff_roc_10d']
    
    data['volume_weighted_price_momentum'] = data['volume'] * (data['close'] / data['close'].shift(5) - 1)
    data['price_volume_consistency'] = np.sign(data['vol_eff_roc_5d']) * np.sign(data['volume_weighted_price_momentum'])
    
    # Volatility Efficiency Regime Duration
    vol_eff_pattern = data['pure_vol_efficiency'] > data['pure_vol_efficiency'].shift(1)
    data['vol_eff_regime_duration'] = vol_eff_pattern.rolling(window=10, min_periods=1).apply(lambda x: len(x) - np.argmin(x[::-1]) if np.any(x) else 0)
    
    # Volume-Volatility Regime Stability
    vol_vol_regime = vol_vol_ratio > vol_vol_ratio.rolling(window=5).mean()
    data['vol_vol_regime_stability'] = vol_vol_regime.rolling(window=10, min_periods=1).apply(lambda x: len(x) - np.argmin(x[::-1]) if np.any(x) else 0)
    
    # Asymmetric Volatility-Price Classification
    upside_vol_eff = (data['high'].rolling(window=3).max() - data['close'].shift(3)) / data['close'].shift(3)
    downside_vol_eff = (data['close'].shift(3) - data['low'].rolling(window=3).min()) / data['close'].shift(3)
    data['vol_asymmetry_ratio'] = upside_vol_eff / downside_vol_eff.replace(0, np.nan)
    
    intraday_price_asymmetry = (data['high'] - data['close']) / (data['close'] - data['low']).replace(0, np.nan)
    data['microstructure_asymmetry_ratio'] = intraday_price_asymmetry / data['gap_vol_size'].replace(0, np.nan)
    
    # Regime Classification
    vol_asymmetry_regime = pd.cut(data['vol_asymmetry_ratio'], 
                                bins=[-np.inf, 0.8, 1.4, np.inf], 
                                labels=['Low', 'Normal', 'High'])
    
    price_efficiency_regime = (data['pure_vol_efficiency'] > 0.6).map({True: 'High', False: 'Low'})
    
    vol_vol_5d_avg = vol_vol_ratio.rolling(window=5).mean()
    volume_price_regime = pd.cut(vol_vol_ratio / vol_vol_5d_avg.replace(0, np.nan),
                               bins=[-np.inf, 0.8, 1.2, np.inf],
                               labels=['Low', 'Normal', 'High'])
    
    price_spread_regime = pd.cut(data['price_spread_proxy'],
                               bins=[-np.inf, 0.01, 0.02, np.inf],
                               labels=['Tight', 'Normal', 'Wide'])
    
    # Regime-Adaptive Volatility-Price Signal Construction
    regime_signals = pd.Series(index=data.index, dtype=float)
    
    # High Asymmetry + High Efficiency Regime
    high_asym_high_eff_mask = (vol_asymmetry_regime == 'High') & (price_efficiency_regime == 'High')
    gap_vol_reversion = -data['gap_vol_size']
    vol_momentum_weight = abs(data['vol_eff_momentum_divergence'])
    regime_signals[high_asym_high_eff_mask] = (gap_vol_reversion * data['price_volume_consistency'] * vol_momentum_weight)[high_asym_high_eff_mask]
    
    # Normal Asymmetry + Low Efficiency Regime
    normal_asym_low_eff_mask = (vol_asymmetry_regime == 'Normal') & (price_efficiency_regime == 'Low')
    price_breakout_readiness = data['pure_vol_efficiency']
    volume_price_concentration = vol_vol_ratio / vol_vol_5d_avg.replace(0, np.nan)
    price_divergence_strength = abs(data['vol_eff_momentum_divergence'])
    regime_signals[normal_asym_low_eff_mask] = (price_breakout_readiness * volume_price_concentration * price_divergence_strength)[normal_asym_low_eff_mask]
    
    # Low Asymmetry + High Efficiency Regime
    low_asym_high_eff_mask = (vol_asymmetry_regime == 'Low') & (price_efficiency_regime == 'High')
    price_acceleration = data['vol_eff_momentum_divergence']
    intraday_price_confirmation = abs(data['intraday_vol_position'])
    volume_price_enhancement = data['volume_weighted_price_momentum']
    regime_signals[low_asym_high_eff_mask] = (price_acceleration * intraday_price_confirmation * volume_price_enhancement)[low_asym_high_eff_mask]
    
    # Fill remaining regimes with default signal
    remaining_mask = regime_signals.isna()
    regime_signals[remaining_mask] = (data['vol_eff_momentum_divergence'] * data['pure_vol_efficiency'] * data['price_volume_consistency'])[remaining_mask]
    
    # Dynamic Volatility-Price Signal Enhancement
    # Volume-Price Spike & Efficiency Integration
    volume_price_spike = (vol_vol_ratio > 1.5 * vol_vol_5d_avg).astype(int)
    price_efficiency_spike_multiplier = 1 + 0.3 * volume_price_spike * data['pure_vol_efficiency']
    spike_enhanced_signal = regime_signals * price_efficiency_spike_multiplier
    
    # Intraday Price Position & Spread Adjustment
    price_directional_bias = data['intraday_vol_position']
    price_spread_adjustment = 1 - data['price_spread_proxy']
    position_spread_adjusted_signal = spike_enhanced_signal * (1 + 0.2 * price_directional_bias) * price_spread_adjustment
    
    # Volatility Correlation Convergence & Regime Filter
    volatility_correlation_regime_filter = 1 + data['vol_eff_regime_shift']
    price_regime_persistence_filter = data['vol_eff_regime_duration'] / 10
    convergence_filtered_signal = position_spread_adjusted_signal * volatility_correlation_regime_filter * (1 + 0.1 * price_regime_persistence_filter)
    
    # Final Volatility-Price Alpha Construction
    # Volatility Asymmetry Scaling
    asymmetry_scaled_signal = convergence_filtered_signal / (data['vol_asymmetry_ratio'] + 0.0001)
    
    # Volatility-Price Consistency Verification
    price_volume_alignment = np.sign(data['vol_eff_roc_5d']) * np.sign(data['price_volume_consistency'])
    price_regime_persistence_score = data['vol_eff_regime_duration'] * data['vol_vol_regime_stability']
    consistency_enhanced_signal = asymmetry_scaled_signal * (1 + 0.1 * price_volume_alignment) * (1 + 0.05 * price_regime_persistence_score)
    
    # Final smoothing and normalization
    final_alpha = consistency_enhanced_signal.rolling(window=5).mean()
    final_alpha = (final_alpha - final_alpha.rolling(window=20).mean()) / final_alpha.rolling(window=20).std()
    
    return final_alpha
