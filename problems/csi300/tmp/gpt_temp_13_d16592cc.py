import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize all intermediate columns
    data['Micro_Transmission'] = 0.0
    data['Meso_Transmission'] = 0.0
    data['Macro_Transmission'] = 0.0
    data['Morning_Fractal_Pressure'] = 0.0
    data['Afternoon_Fractal_Pressure'] = 0.0
    data['Fractal_Pressure_Differential'] = 0.0
    data['Micro_Momentum_Asymmetry'] = 0.0
    data['Volume_Concentration_Asymmetry'] = 0.0
    data['Volume_Micro_Vol'] = 0.0
    data['Volume_Meso_Vol'] = 0.0
    data['Volume_Macro_Vol'] = 0.0
    data['Volume_Volatility_Divergence'] = 0.0
    data['Pattern_Quality_Asymmetry'] = 0.0
    data['Volatility_Regime_Shift'] = 0.0
    data['Fractal_Regime_Alignment'] = 0.0
    data['Fractal_Transmission_Divergence'] = 0.0
    data['Short_term_Divergence'] = 0.0
    data['Medium_term_Divergence'] = 0.0
    data['Multi_Scale_Divergence_Alignment'] = 0.0
    
    # Calculate rolling windows
    data['High_3d'] = data['high'].rolling(window=3, min_periods=1).max()
    data['Low_3d'] = data['low'].rolling(window=3, min_periods=1).min()
    data['High_8d'] = data['high'].rolling(window=8, min_periods=1).max()
    data['Low_8d'] = data['low'].rolling(window=8, min_periods=1).min()
    data['High_5d'] = data['high'].rolling(window=5, min_periods=1).max()
    data['Low_5d'] = data['low'].rolling(window=5, min_periods=1).min()
    
    for i in range(len(data)):
        if i < 8:  # Skip early rows that don't have enough history
            continue
            
        # Multi-Fractal Volatility Transmission
        max_oc = max(data['open'].iloc[i], data['close'].iloc[i])
        min_oc = min(data['open'].iloc[i], data['close'].iloc[i])
        
        # Micro Transmission
        micro_trans = ((data['high'].iloc[i] - max_oc) / (min_oc - data['low'].iloc[i] + 0.001)) * \
                     ((data['close'].iloc[i] - data['open'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i] + 0.001))
        
        # Meso Transmission
        meso_trans = ((data['High_3d'].iloc[i] - max_oc) / (min_oc - data['Low_3d'].iloc[i] + 0.001)) * \
                    ((data['close'].iloc[i] - data['open'].iloc[i]) / (data['High_3d'].iloc[i] - data['Low_3d'].iloc[i] + 0.001))
        
        # Macro Transmission
        macro_trans = ((data['High_8d'].iloc[i] - max_oc) / (min_oc - data['Low_8d'].iloc[i] + 0.001)) * \
                     ((data['close'].iloc[i] - data['open'].iloc[i]) / (data['High_8d'].iloc[i] - data['Low_8d'].iloc[i] + 0.001))
        
        # Fractal Transmission Divergence
        fractal_trans_div = (micro_trans - meso_trans) * (meso_trans - macro_trans) * (micro_trans - macro_trans)
        
        # Fractal Momentum Asymmetry
        morning_pressure = (data['high'].iloc[i] - data['open'].iloc[i]) * (data['close'].iloc[i] - data['low'].iloc[i])
        afternoon_pressure = (data['open'].iloc[i] - data['low'].iloc[i]) * (data['high'].iloc[i] - data['close'].iloc[i])
        pressure_diff = morning_pressure - afternoon_pressure
        
        micro_momentum_asym = ((data['close'].iloc[i] - data['open'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i] + 0.001)) ** 2 * pressure_diff
        
        # Volume-Volatility Divergence
        vol_conc_asym = data['volume'].iloc[i] * pressure_diff
        
        # Volume volatility calculations
        vol_micro = abs(data['volume'].iloc[i] - data['volume'].iloc[i-1]) / (data['volume'].iloc[i-1] + 0.001) if i >= 1 else 0
        vol_meso = abs(data['volume'].iloc[i] - data['volume'].iloc[i-3]) / (data['volume'].iloc[i-3] + 0.001) if i >= 3 else 0
        vol_macro = abs(data['volume'].iloc[i] - data['volume'].iloc[i-8]) / (data['volume'].iloc[i-8] + 0.001) if i >= 8 else 0
        
        vol_vol_div = (vol_micro - vol_meso) * (vol_meso - vol_macro) * (vol_micro - vol_macro)
        
        # Fractal Quality & Regime Patterns
        pattern_qual_asym = 1.0
        if i >= 8:
            term1 = ((data['close'].iloc[i] - data['close'].iloc[i-1]) / (data['high'].iloc[i] - data['low'].iloc[i] + 0.001)) * \
                   ((data['close'].iloc[i] - data['close'].iloc[i-2]) / (data['high'].iloc[i-2] - data['low'].iloc[i-2] + 0.001))
            
            term2 = ((data['close'].iloc[i] - data['close'].iloc[i-5]) / (data['High_5d'].iloc[i] - data['Low_5d'].iloc[i] + 0.001)) * \
                   ((data['close'].iloc[i] - data['close'].iloc[i-8]) / (data['High_8d'].iloc[i] - data['Low_8d'].iloc[i] + 0.001))
            
            pattern_qual_asym = term1 * term2 * pressure_diff
        
        vol_regime_shift = (micro_trans - meso_trans) * (vol_micro - vol_meso)
        fractal_regime_align = pattern_qual_asym * vol_regime_shift * np.sign(pressure_diff) if pressure_diff != 0 else 0
        
        # Multi-Scale Divergence Integration
        short_term_div = micro_momentum_asym * vol_conc_asym * fractal_trans_div
        medium_term_div = vol_vol_div * pattern_qual_asym
        
        multi_scale_align = np.sign(short_term_div) * np.sign(medium_term_div) * np.sign(fractal_trans_div) if short_term_div != 0 and medium_term_div != 0 and fractal_trans_div != 0 else 0
        
        # Regime-Based Alpha Construction
        high_vol_regime = (micro_trans > meso_trans) and (vol_micro > vol_meso)
        
        if high_vol_regime:
            vol_momentum = fractal_trans_div * micro_momentum_asym
            vol_impact = vol_vol_div * vol_conc_asym
            regime_signal = vol_momentum * vol_impact * np.sign(micro_momentum_asym) if micro_momentum_asym != 0 else 0
        else:
            mean_reversion = -fractal_trans_div * vol_vol_div
            breakout_potential = ((data['close'].iloc[i] - data['open'].iloc[i]) / (micro_trans + 0.001)) * vol_conc_asym
            regime_signal = mean_reversion * breakout_potential * np.sign(vol_conc_asym) if vol_conc_asym != 0 else 0
        
        regime_integrated = regime_signal if high_vol_regime or (not high_vol_regime) else 1
        
        # Final Alpha Factor
        alpha = regime_integrated * fractal_regime_align * multi_scale_align * (data['close'].iloc[i] - data['open'].iloc[i])
        
        # Store values
        data.loc[data.index[i], 'Micro_Transmission'] = micro_trans
        data.loc[data.index[i], 'Meso_Transmission'] = meso_trans
        data.loc[data.index[i], 'Macro_Transmission'] = macro_trans
        data.loc[data.index[i], 'Morning_Fractal_Pressure'] = morning_pressure
        data.loc[data.index[i], 'Afternoon_Fractal_Pressure'] = afternoon_pressure
        data.loc[data.index[i], 'Fractal_Pressure_Differential'] = pressure_diff
        data.loc[data.index[i], 'Micro_Momentum_Asymmetry'] = micro_momentum_asym
        data.loc[data.index[i], 'Volume_Concentration_Asymmetry'] = vol_conc_asym
        data.loc[data.index[i], 'Volume_Micro_Vol'] = vol_micro
        data.loc[data.index[i], 'Volume_Meso_Vol'] = vol_meso
        data.loc[data.index[i], 'Volume_Macro_Vol'] = vol_macro
        data.loc[data.index[i], 'Volume_Volatility_Divergence'] = vol_vol_div
        data.loc[data.index[i], 'Pattern_Quality_Asymmetry'] = pattern_qual_asym
        data.loc[data.index[i], 'Volatility_Regime_Shift'] = vol_regime_shift
        data.loc[data.index[i], 'Fractal_Regime_Alignment'] = fractal_regime_align
        data.loc[data.index[i], 'Fractal_Transmission_Divergence'] = fractal_trans_div
        data.loc[data.index[i], 'Short_term_Divergence'] = short_term_div
        data.loc[data.index[i], 'Medium_term_Divergence'] = medium_term_div
        data.loc[data.index[i], 'Multi_Scale_Divergence_Alignment'] = multi_scale_align
        data.loc[data.index[i], 'Fractal_Volatility_Momentum_Divergence_Alpha'] = alpha
    
    # Return only the final alpha factor series
    return data['Fractal_Volatility_Momentum_Divergence_Alpha']
