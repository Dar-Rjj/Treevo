import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility-Entropy Regime Synthesis alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize all components with NaN
    data['gap_vol_entropy'] = np.nan
    data['intraday_range_eff'] = np.nan
    data['true_range_entropy_mom'] = np.nan
    data['vol_intensity_asym'] = np.nan
    data['amount_conc_shift'] = np.nan
    data['vol_spike_entropy'] = np.nan
    data['short_term_eff_mom'] = np.nan
    data['medium_term_eff_ratio'] = np.nan
    data['efficiency_regime_div'] = np.nan
    data['high_vol_entropy_regime'] = np.nan
    data['low_vol_entropy_regime'] = np.nan
    data['transition_entropy_regime'] = np.nan
    data['high_vol_entropy_strength'] = np.nan
    data['low_vol_entropy_strength'] = np.nan
    data['transition_entropy_strength'] = np.nan
    data['cross_regime_entropy_val'] = np.nan
    data['multi_scale_eff_conf'] = np.nan
    data['vol_entropy_alpha'] = np.nan
    
    # Calculate components with proper lag handling
    for i in range(5, len(data)):
        # Entropy-Enhanced Volatility Structure
        if i >= 1:
            gap_vol = (data['open'].iloc[i] - data['close'].iloc[i-1]) / (data['high'].iloc[i-1] - data['low'].iloc[i-1] + 1e-8)
            vol_ratio = data['volume'].iloc[i] / (data['volume'].iloc[i-1] + 1e-8)
            data.loc[data.index[i], 'gap_vol_entropy'] = gap_vol * vol_ratio
            
            intraday_range = (data['close'].iloc[i] - data['open'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i] + 1e-8)
            data.loc[data.index[i], 'intraday_range_eff'] = intraday_range
            
            if i >= 2:
                true_range_ratio = (data['high'].iloc[i] - data['low'].iloc[i]) / (data['high'].iloc[i-1] - data['low'].iloc[i-1] + 1e-8)
                price_sign = np.sign(data['close'].iloc[i] - data['close'].iloc[i-1])
                data.loc[data.index[i], 'true_range_entropy_mom'] = true_range_ratio * price_sign * intraday_range
        
        # Volume-Pressure Regime Detection
        if i >= 1:
            vol_intensity = data['volume'].iloc[i] / (data['volume'].iloc[i-1] + 1e-8)
            price_sign_vol = np.sign(data['close'].iloc[i] - data['close'].iloc[i-1])
            data.loc[data.index[i], 'vol_intensity_asym'] = vol_intensity * price_sign_vol
            
            amount_conc_current = data['amount'].iloc[i] / (data['volume'].iloc[i] + 1e-8)
            amount_conc_prev = data['amount'].iloc[i-1] / (data['volume'].iloc[i-1] + 1e-8)
            data.loc[data.index[i], 'amount_conc_shift'] = amount_conc_current - amount_conc_prev
            
            if i >= 2:
                vol_spike = (data['volume'].iloc[i] / (data['volume'].iloc[i-1] + 1e-8)) - (data['volume'].iloc[i-1] / (data['volume'].iloc[i-2] + 1e-8))
                open_close_sign = np.sign(data['close'].iloc[i] - data['open'].iloc[i])
                data.loc[data.index[i], 'vol_spike_entropy'] = vol_spike * open_close_sign
        
        # Multi-Timeframe Efficiency Synthesis
        if i >= 1:
            short_term_eff = intraday_range * vol_intensity
            data.loc[data.index[i], 'short_term_eff_mom'] = short_term_eff
            
            if i >= 5:
                medium_term_price = data['close'].iloc[i] - data['close'].iloc[i-3]
                medium_term_range = (data['high'].iloc[i] - data['low'].iloc[i] + 
                                   data['high'].iloc[i-1] - data['low'].iloc[i-1] + 
                                   data['high'].iloc[i-2] - data['low'].iloc[i-2] + 1e-8)
                medium_term_vol = (data['volume'].iloc[i] + data['volume'].iloc[i-1] + data['volume'].iloc[i-2]) / (
                    data['volume'].iloc[i-3] + data['volume'].iloc[i-4] + data['volume'].iloc[i-5] + 1e-8)
                data.loc[data.index[i], 'medium_term_eff_ratio'] = (medium_term_price / medium_term_range) * medium_term_vol
                
                if short_term_eff != 0:
                    data.loc[data.index[i], 'efficiency_regime_div'] = short_term_eff / (data['medium_term_eff_ratio'].iloc[i] + 1e-8) * data['vol_spike_entropy'].iloc[i]
        
        # Entropy-Volatility Regime Indicators
        if i >= 2:
            data.loc[data.index[i], 'high_vol_entropy_regime'] = (
                data['true_range_entropy_mom'].iloc[i] * 
                data['vol_spike_entropy'].iloc[i] * 
                data['gap_vol_entropy'].iloc[i]
            )
            
            data.loc[data.index[i], 'low_vol_entropy_regime'] = (
                data['intraday_range_eff'].iloc[i] * 
                data['amount_conc_shift'].iloc[i] * 
                data['vol_intensity_asym'].iloc[i]
            )
            
            data.loc[data.index[i], 'transition_entropy_regime'] = (
                data['efficiency_regime_div'].iloc[i] * 
                data['gap_vol_entropy'].iloc[i] * 
                data['vol_intensity_asym'].iloc[i]
            )
        
        # Regime Strength Synthesis
        if i >= 2:
            data.loc[data.index[i], 'high_vol_entropy_strength'] = (
                data['high_vol_entropy_regime'].iloc[i] * 
                data['vol_spike_entropy'].iloc[i] * 
                data['true_range_entropy_mom'].iloc[i]
            )
            
            data.loc[data.index[i], 'low_vol_entropy_strength'] = (
                data['low_vol_entropy_regime'].iloc[i] * 
                data['amount_conc_shift'].iloc[i] * 
                data['intraday_range_eff'].iloc[i]
            )
            
            data.loc[data.index[i], 'transition_entropy_strength'] = (
                data['transition_entropy_regime'].iloc[i] * 
                data['efficiency_regime_div'].iloc[i] * 
                data['vol_intensity_asym'].iloc[i]
            )
        
        # Final Alpha Generation
        if i >= 2:
            data.loc[data.index[i], 'cross_regime_entropy_val'] = (
                data['high_vol_entropy_strength'].iloc[i] * 
                data['low_vol_entropy_strength'].iloc[i] * 
                data['transition_entropy_strength'].iloc[i]
            )
            
            data.loc[data.index[i], 'multi_scale_eff_conf'] = (
                data['cross_regime_entropy_val'].iloc[i] * 
                data['efficiency_regime_div'].iloc[i] * 
                data['vol_spike_entropy'].iloc[i]
            )
            
            data.loc[data.index[i], 'vol_entropy_alpha'] = (
                data['multi_scale_eff_conf'].iloc[i] * 
                data['high_vol_entropy_regime'].iloc[i] * 
                data['high_vol_entropy_strength'].iloc[i]
            )
    
    # Return the final alpha factor
    return data['vol_entropy_alpha']
