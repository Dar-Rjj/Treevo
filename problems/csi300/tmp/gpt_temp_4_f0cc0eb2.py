import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Pre-calculate rolling windows for efficiency
    for i in range(len(df)):
        if i < 21:  # Need at least 21 periods for calculations
            result.iloc[i] = 0
            continue
            
        # Extract current and historical data
        current_data = df.iloc[:i+1]
        
        # 1. Multi-Scale Fractal Efficiency
        # Price Fractal Efficiency
        # Micro Efficiency
        if i >= 2:
            micro_eff = ((current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) / 
                        (current_data['high'].iloc[i] - current_data['low'].iloc[i] + 1e-8) *
                        (current_data['close'].iloc[i] - current_data['close'].iloc[i-2]) / 
                        (current_data['high'].iloc[i-2] - current_data['low'].iloc[i-2] + 1e-8))
        else:
            micro_eff = 0
            
        # Meso Efficiency
        if i >= 8:
            meso_window_5 = current_data.iloc[i-5:i+1]
            meso_window_8 = current_data.iloc[i-8:i+1]
            meso_eff = ((current_data['close'].iloc[i] - current_data['close'].iloc[i-5]) / 
                       (meso_window_5['high'].max() - meso_window_5['low'].min() + 1e-8) *
                       (current_data['close'].iloc[i] - current_data['close'].iloc[i-8]) / 
                       (meso_window_8['high'].max() - meso_window_8['low'].min() + 1e-8))
        else:
            meso_eff = 0
            
        # Macro Efficiency
        if i >= 21:
            macro_window_13 = current_data.iloc[i-13:i+1]
            macro_window_21 = current_data.iloc[i-21:i+1]
            macro_eff = ((current_data['close'].iloc[i] - current_data['close'].iloc[i-13]) / 
                        (macro_window_13['high'].max() - macro_window_13['low'].min() + 1e-8) *
                        (current_data['close'].iloc[i] - current_data['close'].iloc[i-21]) / 
                        (macro_window_21['high'].max() - macro_window_21['low'].min() + 1e-8))
        else:
            macro_eff = 0
            
        fractal_efficiency_cascade = micro_eff * meso_eff * macro_eff * np.sign(micro_eff - meso_eff + 1e-8)
        
        # Volume Fractal Dynamics
        if i >= 1:
            volume_micro = (current_data['volume'].iloc[i] / (current_data['volume'].iloc[i-1] + 1e-8) * 
                           abs(current_data['volume'].iloc[i] - current_data['volume'].iloc[i-1]) / 
                           (current_data['volume'].iloc[i] + 1e-8))
        else:
            volume_micro = 0
            
        if i >= 5:
            volume_meso = (current_data['volume'].iloc[i] / (current_data['volume'].iloc[i-5] + 1e-8) * 
                          abs(current_data['volume'].iloc[i] - current_data['volume'].iloc[i-5]) / 
                          (current_data['volume'].iloc[i] + 1e-8))
        else:
            volume_meso = 0
            
        if i >= 13:
            volume_macro = (current_data['volume'].iloc[i] / (current_data['volume'].iloc[i-13] + 1e-8) * 
                           abs(current_data['volume'].iloc[i] - current_data['volume'].iloc[i-13]) / 
                           (current_data['volume'].iloc[i] + 1e-8))
        else:
            volume_macro = 0
            
        volume_fractal_cascade = volume_micro * volume_meso * volume_macro * np.sign(volume_micro - volume_meso + 1e-8)
        
        # 2. Gap-Regime Momentum Construction
        # Multi-Scale Gap Analysis
        if i >= 1:
            micro_gap = (current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / (current_data['close'].iloc[i-1] + 1e-8)
        else:
            micro_gap = 0
            
        if i >= 3:
            meso_gap = (current_data['open'].iloc[i] - current_data['close'].iloc[i-3]) / (current_data['close'].iloc[i-3] + 1e-8)
        else:
            meso_gap = 0
            
        if i >= 8:
            macro_gap = (current_data['open'].iloc[i] - current_data['close'].iloc[i-8]) / (current_data['close'].iloc[i-8] + 1e-8)
        else:
            macro_gap = 0
            
        fractal_gap_cascade = micro_gap * meso_gap * macro_gap * np.sign(micro_gap - meso_gap + 1e-8)
        
        # Volume-Regime Momentum
        if i >= 1:
            volume_micro_regime = (current_data['volume'].iloc[i] / (current_data['volume'].iloc[i-1] + 1e-8) * 
                                 (current_data['close'].iloc[i] - current_data['close'].iloc[i-1]))
        else:
            volume_micro_regime = 0
            
        if i >= 3:
            volume_meso_regime = (current_data['volume'].iloc[i] / (current_data['volume'].iloc[i-3] + 1e-8) * 
                                (current_data['close'].iloc[i] - current_data['close'].iloc[i-3]))
        else:
            volume_meso_regime = 0
            
        if i >= 8:
            volume_macro_regime = (current_data['volume'].iloc[i] / (current_data['volume'].iloc[i-8] + 1e-8) * 
                                 (current_data['close'].iloc[i] - current_data['close'].iloc[i-8]))
        else:
            volume_macro_regime = 0
            
        volume_regime_cascade = (volume_micro_regime * volume_meso_regime * volume_macro_regime * 
                               np.sign(volume_micro_regime - volume_meso_regime + 1e-8))
        
        # 3. Absorption Breakout System
        # Liquidity Absorption Dynamics
        if i >= 3:
            volume_concentration = (current_data['volume'].iloc[i] / 
                                  (current_data['volume'].iloc[i-3:i].mean() + 1e-8))
        else:
            volume_concentration = 0
            
        if i >= 5:
            window_5 = current_data.iloc[i-5:i+1]
            liquidity_absorption = ((current_data['high'].iloc[i] - current_data['low'].iloc[i]) * 
                                  current_data['volume'].iloc[i] / 
                                  ((window_5['high'] - window_5['low']) * window_5['volume']).mean() + 1e-8)
        else:
            liquidity_absorption = 0
            
        trade_impact_efficiency = (current_data['amount'].iloc[i] / 
                                 (current_data['volume'].iloc[i] * (current_data['high'].iloc[i] - current_data['low'].iloc[i]) + 1e-8))
        
        # Fractal Range Breakout
        micro_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        
        if i >= 3:
            window_3 = current_data.iloc[i-3:i+1]
            meso_range = window_3['high'].max() - window_3['low'].min()
        else:
            meso_range = micro_range
            
        if i >= 8:
            window_8 = current_data.iloc[i-8:i+1]
            macro_range = window_8['high'].max() - window_8['low'].min()
        else:
            macro_range = meso_range
            
        fractal_range_ratio = (micro_range / (meso_range + 1e-8)) * (meso_range / (macro_range + 1e-8))
        
        # 4. Asymmetric Convergence Framework
        high_low_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        if high_low_range > 0:
            volatility_asymmetry = ((current_data['high'].iloc[i] - current_data['open'].iloc[i]) / high_low_range - 
                                  (current_data['open'].iloc[i] - current_data['low'].iloc[i]) / high_low_range)
            volume_flow_asymmetry = (current_data['amount'].iloc[i] * (current_data['high'].iloc[i] - current_data['close'].iloc[i]) / high_low_range - 
                                   current_data['amount'].iloc[i] * (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / high_low_range)
        else:
            volatility_asymmetry = 0
            volume_flow_asymmetry = 0
            
        signal_convergence = np.sign(volatility_asymmetry + 1e-8) * np.sign(volume_flow_asymmetry + 1e-8)
        
        # 5. Regime-Based Integration
        # Intraday Efficiency
        if high_low_range > 0:
            intraday_efficiency = abs(current_data['close'].iloc[i] - current_data['open'].iloc[i]) / high_low_range
        else:
            intraday_efficiency = 0
            
        # High Efficiency Regime
        if intraday_efficiency > 0.7 and volume_concentration > 1:
            fractal_absorption = fractal_efficiency_cascade * liquidity_absorption
            gap_momentum = fractal_gap_cascade * volume_regime_cascade
            regime_signal = fractal_absorption * gap_momentum * signal_convergence
            
        # Low Efficiency Regime
        else:
            fractal_contrarian = -fractal_efficiency_cascade * volume_fractal_cascade
            range_breakout = fractal_range_ratio * (current_data['close'].iloc[i] - current_data['open'].iloc[i])
            regime_signal = fractal_contrarian * range_breakout * volatility_asymmetry
        
        # Regime Transition Detection
        if i >= 3:
            efficiency_regime_shift = ((intraday_efficiency - 
                                      (abs(current_data['close'].iloc[i-3] - current_data['open'].iloc[i-3]) / 
                                       (current_data['high'].iloc[i-3] - current_data['low'].iloc[i-3] + 1e-8))) * 
                                     (volume_concentration - 1))
        else:
            efficiency_regime_shift = 0
            
        fractal_regime_transition = (np.sign(fractal_efficiency_cascade - volume_fractal_cascade + 1e-8) * 
                                   volatility_asymmetry)
        
        transition_factor = efficiency_regime_shift * fractal_regime_transition
        
        # 6. Hierarchical Alpha Synthesis
        core_efficiency_signal = regime_signal
        absorption_enhanced = core_efficiency_signal * trade_impact_efficiency
        convergence_weighted = absorption_enhanced * abs(signal_convergence) * abs(volatility_asymmetry)
        transition_adjusted = convergence_weighted * transition_factor
        final_alpha = transition_adjusted * np.sign(core_efficiency_signal + 1e-8)
        
        result.iloc[i] = final_alpha
    
    return result
