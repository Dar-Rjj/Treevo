import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize all intermediate columns
    data['Morning_Fractal_Asymmetry'] = np.nan
    data['Afternoon_Fractal_Asymmetry'] = np.nan
    data['Intraday_Fractal_Asymmetry_Ratio'] = np.nan
    data['Volume_Fractal_Direction'] = np.nan
    data['Volume_Fractal_Strength'] = np.nan
    data['Volume_Fractal_Convergence'] = np.nan
    data['Overnight_Fractal_Gap'] = np.nan
    data['Intraday_Fractal_Recovery'] = np.nan
    data['Fractal_Gap_Momentum'] = np.nan
    data['Opening_Fractal_Asymmetry'] = np.nan
    data['Closing_Fractal_Asymmetry'] = np.nan
    data['Asymmetric_Fractal_Gap'] = np.nan
    data['Volume_Fractal_Efficiency'] = np.nan
    data['Volume_Fractal_Acceleration'] = np.nan
    data['Gap_Volume_Fractal_Asymmetry'] = np.nan
    data['Price_Fractal_Divergence_Magnitude'] = np.nan
    data['Volume_Fractal_Divergence'] = np.nan
    data['Session_Fractal_Divergence'] = np.nan
    data['Fractal_Convergence_Divergence'] = np.nan
    data['Short_term_Fractal_Volatility'] = np.nan
    data['Fractal_Volatility_Transition'] = np.nan
    data['Fractal_Volatility_Regime'] = np.nan
    data['Fractal_Asymmetry_Volatility'] = np.nan
    data['Fractal_Convergence_Volatility'] = np.nan
    data['Volatility_Enhanced_Fractal'] = np.nan
    data['Relative_Fractal_Price'] = np.nan
    data['Fractal_Level_Momentum'] = np.nan
    data['Fractal_Asymmetry_Level'] = np.nan
    data['Fractal_Convergence_Level'] = np.nan
    data['Level_Enhanced_Fractal'] = np.nan
    data['Fractal_Momentum_Amplitude'] = np.nan
    data['Persistent_Fractal_Momentum'] = np.nan
    data['Core_Fractal_Asymmetry'] = np.nan
    data['Fractal_Asymmetry_Convergence_Alpha'] = np.nan
    
    for i in range(13, len(data)):
        # Fractal Asymmetry Microstructure
        if i >= 1:
            morning_asym = ((data['high'].iloc[i] - data['open'].iloc[i]) / 
                           (data['open'].iloc[i] - data['low'].iloc[i] + 0.001) * 
                           (data['close'].iloc[i] - data['close'].iloc[i-1]) / 
                           (data['high'].iloc[i] - data['low'].iloc[i] + 0.001))
            data.loc[data.index[i], 'Morning_Fractal_Asymmetry'] = morning_asym
        
        if i >= 2:
            afternoon_asym = ((data['close'].iloc[i] - data['low'].iloc[i]) / 
                             (data['high'].iloc[i] - data['close'].iloc[i] + 0.001) * 
                             (data['close'].iloc[i] - data['close'].iloc[i-2]) / 
                             (data['high'].iloc[i-2] - data['low'].iloc[i-2] + 0.001))
            data.loc[data.index[i], 'Afternoon_Fractal_Asymmetry'] = afternoon_asym
        
        if i >= 2:
            intraday_ratio = (data['Morning_Fractal_Asymmetry'].iloc[i] / 
                             (data['Afternoon_Fractal_Asymmetry'].iloc[i] + 0.001))
            data.loc[data.index[i], 'Intraday_Fractal_Asymmetry_Ratio'] = intraday_ratio
        
        # Volume Fractal Convergence
        if i >= 1:
            vol_dir = (np.sign(data['close'].iloc[i] - data['close'].iloc[i-1]) * 
                      np.sign(data['volume'].iloc[i] / (data['volume'].iloc[i-1] + 0.001) * 
                      abs(data['volume'].iloc[i] - data['volume'].iloc[i-1]) / 
                      (data['volume'].iloc[i] + 0.001)))
            data.loc[data.index[i], 'Volume_Fractal_Direction'] = vol_dir
        
        if i >= 5:
            vol_strength = (abs(data['close'].iloc[i] - data['close'].iloc[i-1]) * 
                           abs(data['volume'].iloc[i] / (data['volume'].iloc[i-5] + 0.001) * 
                           abs(data['volume'].iloc[i] - data['volume'].iloc[i-5]) / 
                           (data['volume'].iloc[i] + 0.001)))
            data.loc[data.index[i], 'Volume_Fractal_Strength'] = vol_strength
        
        if i >= 5:
            vol_conv = (data['Volume_Fractal_Direction'].iloc[i] * 
                       data['Volume_Fractal_Strength'].iloc[i])
            data.loc[data.index[i], 'Volume_Fractal_Convergence'] = vol_conv
        
        # Gap-Enhanced Fractal Asymmetry
        if i >= 5:
            high_5 = data['high'].iloc[i-5:i+1].max()
            low_5 = data['low'].iloc[i-5:i+1].min()
            overnight_gap = ((data['open'].iloc[i] - data['close'].iloc[i-1]) / 
                            (data['close'].iloc[i-1] + 0.001) * 
                            (data['close'].iloc[i] - data['close'].iloc[i-5]) / 
                            (high_5 - low_5 + 0.001))
            data.loc[data.index[i], 'Overnight_Fractal_Gap'] = overnight_gap
        
        if i >= 8:
            high_8 = data['high'].iloc[i-8:i+1].max()
            low_8 = data['low'].iloc[i-8:i+1].min()
            intraday_recovery = ((data['close'].iloc[i] - data['open'].iloc[i]) / 
                               (data['open'].iloc[i] - data['close'].iloc[i-1] + 0.001) * 
                               (data['close'].iloc[i] - data['close'].iloc[i-8]) / 
                               (high_8 - low_8 + 0.001))
            data.loc[data.index[i], 'Intraday_Fractal_Recovery'] = intraday_recovery
        
        if i >= 8:
            gap_momentum = (data['Overnight_Fractal_Gap'].iloc[i] * 
                           data['Intraday_Fractal_Recovery'].iloc[i])
            data.loc[data.index[i], 'Fractal_Gap_Momentum'] = gap_momentum
        
        if i >= 2:
            opening_asym = ((data['high'].iloc[i] - data['open'].iloc[i]) * 
                           (data['open'].iloc[i] - data['low'].iloc[i]) * 
                           (data['close'].iloc[i] - data['close'].iloc[i-1]) / 
                           (data['high'].iloc[i] - data['low'].iloc[i] + 0.001))
            data.loc[data.index[i], 'Opening_Fractal_Asymmetry'] = opening_asym
            
            closing_asym = ((data['close'].iloc[i] - data['low'].iloc[i]) * 
                           (data['high'].iloc[i] - data['close'].iloc[i]) * 
                           (data['close'].iloc[i] - data['close'].iloc[i-2]) / 
                           (data['high'].iloc[i-2] - data['low'].iloc[i-2] + 0.001))
            data.loc[data.index[i], 'Closing_Fractal_Asymmetry'] = closing_asym
        
        if i >= 8:
            asym_gap = (data['Fractal_Gap_Momentum'].iloc[i] * 
                       (data['Opening_Fractal_Asymmetry'].iloc[i] - 
                        data['Closing_Fractal_Asymmetry'].iloc[i]))
            data.loc[data.index[i], 'Asymmetric_Fractal_Gap'] = asym_gap
        
        if i >= 1:
            vol_eff = ((data['close'].iloc[i] - data['close'].iloc[i-1]) / 
                      (data['volume'].iloc[i] + 0.001) * 
                      data['volume'].iloc[i] / (data['volume'].iloc[i-1] + 0.001))
            data.loc[data.index[i], 'Volume_Fractal_Efficiency'] = vol_eff
            
            vol_acc = ((data['close'].iloc[i] - data['open'].iloc[i]) * 
                      data['volume'].iloc[i] * 
                      abs(data['close'].iloc[i] - data['open'].iloc[i]) * 
                      abs(data['volume'].iloc[i] - data['volume'].iloc[i-1]) / 
                      (data['volume'].iloc[i] + 0.001))
            data.loc[data.index[i], 'Volume_Fractal_Acceleration'] = vol_acc
        
        if i >= 8:
            gap_vol_asym = (data['Asymmetric_Fractal_Gap'].iloc[i] * 
                           data['Volume_Fractal_Efficiency'].iloc[i] * 
                           data['Volume_Fractal_Acceleration'].iloc[i])
            data.loc[data.index[i], 'Gap_Volume_Fractal_Asymmetry'] = gap_vol_asym
        
        # Fractal Convergence-Divergence
        if i >= 8:
            price_div_mag = (abs(data['Morning_Fractal_Asymmetry'].iloc[i] - 
                               data['Afternoon_Fractal_Asymmetry'].iloc[i]) * 
                           abs(data['Opening_Fractal_Asymmetry'].iloc[i] - 
                               data['Closing_Fractal_Asymmetry'].iloc[i]))
            data.loc[data.index[i], 'Price_Fractal_Divergence_Magnitude'] = price_div_mag
            
            vol_div = (data['Volume_Fractal_Convergence'].iloc[i] * 
                      data['Price_Fractal_Divergence_Magnitude'].iloc[i])
            data.loc[data.index[i], 'Volume_Fractal_Divergence'] = vol_div
            
            session_div = (data['Fractal_Gap_Momentum'].iloc[i] * 
                          data['Intraday_Fractal_Asymmetry_Ratio'].iloc[i])
            data.loc[data.index[i], 'Session_Fractal_Divergence'] = session_div
            
            conv_div = (data['Volume_Fractal_Divergence'].iloc[i] * 
                       data['Session_Fractal_Divergence'].iloc[i])
            data.loc[data.index[i], 'Fractal_Convergence_Divergence'] = conv_div
        
        # Volatility-Regime Fractal Asymmetry
        if i >= 1:
            short_vol = ((data['high'].iloc[i] - data['low'].iloc[i]) * 
                        (data['close'].iloc[i] - data['close'].iloc[i-1]) / 
                        (data['high'].iloc[i] - data['low'].iloc[i] + 0.001))
            data.loc[data.index[i], 'Short_term_Fractal_Volatility'] = short_vol
        
        if i >= 5:
            vol_trans = ((data['high'].iloc[i] - data['low'].iloc[i]) / 
                        (data['high'].iloc[i-1] - data['low'].iloc[i-1] + 0.001) * 
                        (data['close'].iloc[i] - data['close'].iloc[i-5]) / 
                        (high_5 - low_5 + 0.001))
            data.loc[data.index[i], 'Fractal_Volatility_Transition'] = vol_trans
        
        if i >= 5:
            vol_regime = (data['Short_term_Fractal_Volatility'].iloc[i] * 
                         data['Fractal_Volatility_Transition'].iloc[i])
            data.loc[data.index[i], 'Fractal_Volatility_Regime'] = vol_regime
        
        if i >= 8:
            asym_vol = (data['Intraday_Fractal_Asymmetry_Ratio'].iloc[i] * 
                       data['Fractal_Volatility_Regime'].iloc[i])
            data.loc[data.index[i], 'Fractal_Asymmetry_Volatility'] = asym_vol
            
            conv_vol = (data['Volume_Fractal_Convergence'].iloc[i] * 
                       data['Fractal_Volatility_Transition'].iloc[i])
            data.loc[data.index[i], 'Fractal_Convergence_Volatility'] = conv_vol
            
            vol_enhanced = (data['Fractal_Asymmetry_Volatility'].iloc[i] * 
                           data['Fractal_Convergence_Volatility'].iloc[i])
            data.loc[data.index[i], 'Volatility_Enhanced_Fractal'] = vol_enhanced
        
        # Fractal Momentum Persistence
        if i >= 13:
            high_13 = data['high'].iloc[i-13:i+1].max()
            low_13 = data['low'].iloc[i-13:i+1].min()
            rel_price = (data['close'].iloc[i] / 
                        ((data['close'].iloc[i-1] + data['close'].iloc[i-2] + 
                          data['close'].iloc[i-3] + data['close'].iloc[i-4] + 
                          data['close'].iloc[i-5]) / 5) * 
                        (data['close'].iloc[i] - data['close'].iloc[i-13]) / 
                        (high_13 - low_13 + 0.001))
            data.loc[data.index[i], 'Relative_Fractal_Price'] = rel_price
            
            level_momentum = ((data['close'].iloc[i] / data['close'].iloc[i-5] - 1) * 
                             data['Relative_Fractal_Price'].iloc[i])
            data.loc[data.index[i], 'Fractal_Level_Momentum'] = level_momentum
        
        if i >= 13:
            asym_level = (data['Intraday_Fractal_Asymmetry_Ratio'].iloc[i] * 
                         data['Fractal_Level_Momentum'].iloc[i])
            data.loc[data.index[i], 'Fractal_Asymmetry_Level'] = asym_level
            
            conv_level = (data['Volume_Fractal_Convergence'].iloc[i] * 
                         data['Relative_Fractal_Price'].iloc[i])
            data.loc[data.index[i], 'Fractal_Convergence_Level'] = conv_level
            
            level_enhanced = (data['Fractal_Asymmetry_Level'].iloc[i] * 
                             data['Fractal_Convergence_Level'].iloc[i])
            data.loc[data.index[i], 'Level_Enhanced_Fractal'] = level_enhanced
        
        if i >= 16:
            # Fractal Signal Persistence (last 3 days)
            if i >= 18:
                persistence = 0
                for j in range(1, 4):
                    if (np.sign(data['Level_Enhanced_Fractal'].iloc[i-j]) == 
                        np.sign(data['Level_Enhanced_Fractal'].iloc[i-j-1])):
                        persistence += 1
            else:
                persistence = 1
            
            mom_amplitude = (data['Level_Enhanced_Fractal'].iloc[i] * 
                           abs(data['Gap_Volume_Fractal_Asymmetry'].iloc[i]))
            data.loc[data.index[i], 'Fractal_Momentum_Amplitude'] = mom_amplitude
            
            persistent_momentum = (mom_amplitude * persistence)
            data.loc[data.index[i], 'Persistent_Fractal_Momentum'] = persistent_momentum
        
        # Final Fractal Asymmetry Factor
        if i >= 16:
            core_asym = (data['Persistent_Fractal_Momentum'].iloc[i] * 
                        data['Fractal_Convergence_Divergence'].iloc[i])
            data.loc[data.index[i], 'Core_Fractal_Asymmetry'] = core_asym
            
            final_alpha = (data['Core_Fractal_Asymmetry'].iloc[i] * 
                          data['Volatility_Enhanced_Fractal'].iloc[i])
            data.loc[data.index[i], 'Fractal_Asymmetry_Convergence_Alpha'] = final_alpha
    
    return data['Fractal_Asymmetry_Convergence_Alpha']
