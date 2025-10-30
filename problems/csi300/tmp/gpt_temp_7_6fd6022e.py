import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Range-Volume Convergence Momentum factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Dynamic Range Efficiency Analysis
    # Adaptive True Range
    true_range = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1)),
        'cc': abs(close - close.shift(1))
    }).max(axis=1)
    
    # Efficiency Persistence (5-day window)
    close_diff_sum = close.diff().abs().rolling(window=5).sum()
    true_range_sum = true_range.rolling(window=5).sum()
    efficiency_persistence = close_diff_sum / true_range_sum
    
    # Volatility-Adjusted Efficiency
    close_std = close.rolling(window=5).std()
    close_mean = close.rolling(window=5).mean()
    volatility_ratio = close_std / close_mean
    volatility_adjusted_efficiency = efficiency_persistence / volatility_ratio
    
    # Range Evolution Pattern
    range_change_momentum = ((high - low) / (high.shift(5) - low.shift(5)) - 1)
    range_efficiency = abs(close - close.shift(1)) / (high - low)
    
    # Volume-Price Convergence Dynamics
    # Volume-Weighted Momentum
    price_momentum_5 = close / close.shift(5) - 1
    volume_ratio_5 = volume / volume.shift(5)
    volume_adjusted_return = price_momentum_5 * np.log(volume_ratio_5)
    
    volume_trend_consistency = np.sign(close - close.shift(1)) * np.sign(volume - volume.shift(1))
    
    # Convergence-Divergence Analysis
    price_momentum_10 = close / close.shift(10) - 1
    price_momentum_convergence = price_momentum_5 * price_momentum_10
    
    volume_momentum_5 = volume / volume.shift(5) - 1
    volume_momentum_10 = volume / volume.shift(10) - 1
    volume_momentum_divergence = abs(volume_momentum_5 - volume_momentum_10)
    
    convergence_divergence_ratio = price_momentum_convergence / (volume_momentum_divergence + 1e-8)
    
    # Volume-Range Interaction
    volume_per_unit_range = volume / (high - low + 1e-8)
    volume_range_momentum = (volume_per_unit_range / volume_per_unit_range.shift(5) - 1)
    
    # Close Position Ratio
    close_position_ratio = (close - low) / (high - low + 1e-8)
    
    # Adaptive Regime Classification
    efficiency_regime = pd.cut(efficiency_persistence, 
                              bins=[-np.inf, 0.3, 0.7, np.inf], 
                              labels=['low', 'medium', 'high'])
    
    range_dynamics_regime = pd.cut(range_change_momentum, 
                                  bins=[-np.inf, -0.1, 0.1, np.inf], 
                                  labels=['compressing', 'stable', 'expanding'])
    
    convergence_regime = pd.cut(convergence_divergence_ratio, 
                               bins=[-np.inf, 0.8, 1.5, np.inf], 
                               labels=['divergence', 'moderate', 'strong'])
    
    # Regime-Adaptive Signal Generation
    for i in range(20, len(df)):
        if pd.isna(efficiency_persistence.iloc[i]) or pd.isna(range_change_momentum.iloc[i]) or pd.isna(convergence_divergence_ratio.iloc[i]):
            continue
            
        eff_reg = efficiency_regime.iloc[i]
        range_reg = range_dynamics_regime.iloc[i]
        conv_reg = convergence_regime.iloc[i]
        
        # High Efficiency + Expanding Range + Strong Convergence
        if eff_reg == 'high' and range_reg == 'expanding' and conv_reg == 'strong':
            primary = efficiency_persistence.iloc[i] * volume_adjusted_return.iloc[i] * 0.5
            secondary = convergence_divergence_ratio.iloc[i] * range_efficiency.iloc[i] * 0.3
            tertiary = volume_range_momentum.iloc[i] * volume_trend_consistency.iloc[i] * 0.2
            signal = primary + secondary + tertiary
            
        # Medium Efficiency + Stable Range + Moderate Convergence
        elif eff_reg == 'medium' and range_reg == 'stable' and conv_reg == 'moderate':
            comp1 = volatility_adjusted_efficiency.iloc[i] * volume_per_unit_range.iloc[i] * 0.4
            comp2 = range_efficiency.iloc[i] * close_position_ratio.iloc[i] * 0.3
            comp3 = volume_trend_consistency.iloc[i] * price_momentum_convergence.iloc[i] * 0.3
            signal = comp1 + comp2 + comp3
            
        # Low Efficiency + Compressing Range + Divergence
        elif eff_reg == 'low' and range_reg == 'compressing' and conv_reg == 'divergence':
            comp1 = range_change_momentum.iloc[i] * volume_momentum_divergence.iloc[i] * 0.5
            comp2 = range_efficiency.iloc[i] * volume_per_unit_range.iloc[i] * 0.3
            comp3 = close_position_ratio.iloc[i] * volume_trend_consistency.iloc[i] * 0.2
            signal = comp1 + comp2 + comp3
            
        # Mixed Regime Combinations
        else:
            # Efficiency-Volume Priority
            eff_vol = efficiency_persistence.iloc[i] * volume_adjusted_return.iloc[i]
            # Range-Convergence Priority
            range_conv = range_efficiency.iloc[i] * convergence_divergence_ratio.iloc[i]
            # Volume-Range Priority
            vol_range = volume_range_momentum.iloc[i] * range_change_momentum.iloc[i]
            
            signal = (eff_vol + range_conv + vol_range) / 3
        
        # Signal Persistence Filter (3-day confirmation)
        if i >= 3:
            prev_signals = [result.iloc[i-1], result.iloc[i-2], result.iloc[i-3]]
            if not all(pd.isna(x) for x in prev_signals):
                valid_signals = [x for x in prev_signals if not pd.isna(x)]
                if len(valid_signals) >= 2:
                    signal_consistency = np.mean([np.sign(x) for x in valid_signals[-2:]])
                    if abs(signal_consistency) >= 0.5:
                        result.iloc[i] = signal
                    else:
                        result.iloc[i] = signal * 0.5  # Reduce signal strength for inconsistency
                else:
                    result.iloc[i] = signal
            else:
                result.iloc[i] = signal
        else:
            result.iloc[i] = signal
    
    return result
