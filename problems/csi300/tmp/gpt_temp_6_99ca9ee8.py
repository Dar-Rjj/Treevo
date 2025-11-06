import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(30, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Fractal Momentum Dynamics
        # Micro Fractal Momentum
        if i >= 1:
            close_diff_micro = current_data['close'].iloc[i] - current_data['close'].iloc[i-1]
            micro_momentum = (np.sign(close_diff_micro) * 
                            (abs(close_diff_micro) ** 2.5) / 
                            (current_data['close'].iloc[i-1] + 0.001) * 
                            ((current_data['high'].iloc[i] - current_data['low'].iloc[i]) ** 1.8) / 
                            (current_data['volume'].iloc[i] + 1))
        else:
            micro_momentum = 0
        
        # Meso Fractal Momentum
        if i >= 5:
            close_diff_meso = current_data['close'].iloc[i] - current_data['close'].iloc[i-5]
            high_range_meso = current_data['high'].iloc[i-5:i+1].max() - current_data['low'].iloc[i-5:i+1].min()
            meso_momentum = (np.sign(close_diff_meso) * 
                           (abs(close_diff_meso) ** 2.2) / 
                           (current_data['close'].iloc[i-5] + 0.001) * 
                           (high_range_meso ** 1.4) / 
                           (current_data['volume'].iloc[i] + 1))
        else:
            meso_momentum = 0
        
        # Macro Fractal Momentum
        if i >= 10:
            close_diff_macro = current_data['close'].iloc[i] - current_data['close'].iloc[i-10]
            high_range_macro = current_data['high'].iloc[i-10:i+1].max() - current_data['low'].iloc[i-10:i+1].min()
            macro_momentum = (np.sign(close_diff_macro) * 
                            (abs(close_diff_macro) ** 1.8) / 
                            (current_data['close'].iloc[i-10] + 0.001) * 
                            (high_range_macro ** 1.1) / 
                            (current_data['volume'].iloc[i] + 1))
        else:
            macro_momentum = 0
        
        # Fractal Momentum Convergence
        if i >= 2 and micro_momentum != 0 and meso_momentum != 0 and macro_momentum != 0:
            momentum_convergence = ((micro_momentum * meso_momentum * macro_momentum) ** (1/3) * 
                                  np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-2]))
        else:
            momentum_convergence = 0
        
        # Fractal Volume-Price Fractality
        # Volume Fractal Dimension
        if i >= 3:
            vol_diff = np.log(current_data['volume'].iloc[i] + 1) - np.log(current_data['volume'].iloc[i-3] + 1)
            range_diff = (np.log(current_data['high'].iloc[i] - current_data['low'].iloc[i] + 0.001) - 
                         np.log(current_data['high'].iloc[i-3] - current_data['low'].iloc[i-3] + 0.001))
            if abs(range_diff) > 1e-10:
                vol_fractal_dim = ((vol_diff ** 2) / (range_diff ** 2) * 
                                 np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]))
            else:
                vol_fractal_dim = 0
        else:
            vol_fractal_dim = 0
        
        # Price Fractal Dimension
        if i >= 5:
            range_diff_price = (np.log(current_data['high'].iloc[i] - current_data['low'].iloc[i] + 0.001) - 
                              np.log(current_data['high'].iloc[i-5] - current_data['low'].iloc[i-5] + 0.001))
            vol_diff_price = np.log(current_data['volume'].iloc[i] + 1) - np.log(current_data['volume'].iloc[i-5] + 1)
            if abs(vol_diff_price) > 1e-10:
                price_fractal_dim = ((range_diff_price ** 2) / (vol_diff_price ** 2) * 
                                   np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-2]))
            else:
                price_fractal_dim = 0
        else:
            price_fractal_dim = 0
        
        # Fractal Pressure Differential
        high_low_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        if high_low_range > 1e-10 and i >= 2:
            pressure_diff = (((current_data['high'].iloc[i] - current_data['close'].iloc[i]) ** 2.5 - 
                            (current_data['close'].iloc[i] - current_data['low'].iloc[i]) ** 2.5) / 
                           high_low_range * 
                           (current_data['volume'].iloc[i] ** 1.3 - current_data['volume'].iloc[i-2] ** 1.3))
        else:
            pressure_diff = 0
        
        # Fractal Regime Signal
        regime_signal = (np.sign(vol_fractal_dim) * np.sign(price_fractal_dim) * 
                        pressure_diff * (current_data['close'].iloc[i] - current_data['open'].iloc[i]) ** 1.7)
        
        # Fractal Volatility Scaling
        # Micro Volatility Fractal
        if i >= 3:
            micro_vol_diff = ((current_data['high'].iloc[i] - current_data['low'].iloc[i]) ** 1.7 - 
                            (current_data['high'].iloc[i-3] - current_data['low'].iloc[i-3]) ** 1.7)
            denom_micro = current_data['high'].iloc[i-3] - current_data['low'].iloc[i-3] + 0.001
            if abs(denom_micro) > 1e-10:
                micro_vol_fractal = (micro_vol_diff / denom_micro * 
                                   np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) * 
                                   (current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) ** 2.3)
            else:
                micro_vol_fractal = 0
        else:
            micro_vol_fractal = 0
        
        # Meso Volatility Fractal
        if i >= 10:
            current_range_meso = current_data['high'].iloc[i-5:i+1].max() - current_data['low'].iloc[i-5:i+1].min()
            prev_range_meso = current_data['high'].iloc[i-10:i-5].max() - current_data['low'].iloc[i-10:i-5].min()
            meso_vol_diff = (current_range_meso ** 1.5 - prev_range_meso ** 1.5)
            if prev_range_meso > 1e-10:
                meso_vol_fractal = (meso_vol_diff / (prev_range_meso + 0.001) * 
                                  (np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-5]) ** 2.1))
            else:
                meso_vol_fractal = 0
        else:
            meso_vol_fractal = 0
        
        # Macro Volatility Fractal
        if i >= 30:
            current_range_macro = current_data['high'].iloc[i-15:i+1].max() - current_data['low'].iloc[i-15:i+1].min()
            prev_range_macro = current_data['high'].iloc[i-30:i-15].max() - current_data['low'].iloc[i-30:i-15].min()
            macro_vol_diff = (current_range_macro ** 1.3 - prev_range_macro ** 1.3)
            if prev_range_macro > 1e-10:
                macro_vol_fractal = (macro_vol_diff / (prev_range_macro + 0.001) * 
                                   (np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-10]) ** 1.8))
            else:
                macro_vol_fractal = 0
        else:
            macro_vol_fractal = 0
        
        # Volatility Fractal Signal
        if (micro_vol_fractal != 0 and meso_vol_fractal != 0 and macro_vol_fractal != 0 and i >= 3):
            vol_fractal_signal = ((micro_vol_fractal * meso_vol_fractal * macro_vol_fractal) ** (1/3) * 
                                (current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) ** 1.5)
        else:
            vol_fractal_signal = 0
        
        # Fractal Persistence Patterns
        # Fractal Momentum Persistence
        momentum_persistence = 1
        if i >= 3:
            for lag in [1, 2, 3]:
                if i - lag >= 0 and micro_momentum != 0:
                    if np.sign(micro_momentum) == np.sign(micro_momentum):
                        momentum_persistence += 1
        
        if i >= 5:
            for lag in [1, 3, 5]:
                if i - lag >= 0 and meso_momentum != 0:
                    if np.sign(meso_momentum) == np.sign(meso_momentum):
                        momentum_persistence += 1
        
        # Fractal Volume Persistence
        volume_persistence = 1
        if i >= 4:
            for lag in [1, 2, 4]:
                if i - lag >= 0 and vol_fractal_dim != 0:
                    if np.sign(vol_fractal_dim) == np.sign(vol_fractal_dim):
                        volume_persistence += 1
        
        if i >= 6:
            for lag in [1, 3, 6]:
                if i - lag >= 0 and price_fractal_dim != 0:
                    if np.sign(price_fractal_dim) == np.sign(price_fractal_dim):
                        volume_persistence += 1
        
        # Fractal Volatility Persistence
        volatility_persistence = 1
        if i >= 3:
            for lag in [1, 2, 3]:
                if i - lag >= 0 and micro_vol_fractal != 0:
                    if np.sign(micro_vol_fractal) == np.sign(micro_vol_fractal):
                        volatility_persistence += 1
        
        if i >= 5:
            for lag in [1, 3, 5]:
                if i - lag >= 0 and meso_vol_fractal != 0:
                    if np.sign(meso_vol_fractal) == np.sign(meso_vol_fractal):
                        volatility_persistence += 1
        
        # Fractal Persistence Score
        if i >= 4:
            persistence_score = ((momentum_persistence * volume_persistence * volatility_persistence) ** (1/3) * 
                               (np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-4]) ** 2.2))
        else:
            persistence_score = 1
        
        # Alpha Synthesis
        # Core Fractal Components
        if (momentum_convergence != 0 and regime_signal != 0 and vol_fractal_signal != 0):
            core_components = ((momentum_convergence * regime_signal * vol_fractal_signal) ** (1/3) * 
                             (current_data['close'].iloc[i] - current_data['open'].iloc[i]) ** 1.8)
        else:
            core_components = 0
        
        # Fractal Persistence Multiplier
        persistence_multiplier = persistence_score * (1 + abs(pressure_diff) ** 1.4)
        
        # Final Alpha
        if core_components != 0:
            final_alpha = (core_components * persistence_multiplier * 
                         (current_data['high'].iloc[i] - current_data['low'].iloc[i]) ** 1.2 / 
                         (current_data['volume'].iloc[i] + 1))
        else:
            final_alpha = 0
        
        result.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
