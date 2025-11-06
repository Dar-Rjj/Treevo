import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate all required components
    for i in range(6, len(df)):
        current_data = df.iloc[:i+1]
        
        # Entropic Fractal Spectrum
        # Quantum Fractal Efficiency
        if i >= 5:
            close_t = current_data['close'].iloc[i]
            close_t_minus_5 = current_data['close'].iloc[i-5]
            high_range_5 = current_data['high'].iloc[i-4:i+1].max()
            low_range_5 = current_data['low'].iloc[i-4:i+1].min()
            open_t = current_data['open'].iloc[i]
            close_t_minus_1 = current_data['close'].iloc[i-1]
            
            fractal_efficiency = (abs(close_t - close_t_minus_5) / (high_range_5 - low_range_5 + 1e-8) * 
                                abs(open_t - close_t_minus_1) / (close_t_minus_1 + 1e-8))
        else:
            fractal_efficiency = 0
        
        # Bidirectional Fractal Momentum
        if i >= 10:
            close_t_minus_2 = current_data['close'].iloc[i-2]
            volume_t = current_data['volume'].iloc[i]
            volume_t_minus_1 = current_data['volume'].iloc[i-1]
            
            # 10-day efficiency
            close_t_minus_10 = current_data['close'].iloc[i-10]
            high_range_10 = current_data['high'].iloc[i-9:i+1].max()
            low_range_10 = current_data['low'].iloc[i-9:i+1].min()
            efficiency_10 = abs(close_t - close_t_minus_10) / (high_range_10 - low_range_10 + 1e-8)
            
            # 5-day efficiency
            close_t_minus_5 = current_data['close'].iloc[i-5]
            high_range_5 = current_data['high'].iloc[i-4:i+1].max()
            low_range_5 = current_data['low'].iloc[i-4:i+1].min()
            efficiency_5 = abs(close_t - close_t_minus_5) / (high_range_5 - low_range_5 + 1e-8)
            
            bidirectional_momentum = ((efficiency_10 - efficiency_5) * 
                                    np.sign(close_t - close_t_minus_2) * 
                                    volume_t / (volume_t_minus_1 + 1e-8))
        else:
            bidirectional_momentum = 0
        
        # Quantum Price Compression
        if i >= 6:
            high_range_3 = current_data['high'].iloc[i-2:i+1].max()
            low_range_3 = current_data['low'].iloc[i-2:i+1].min()
            high_range_6 = current_data['high'].iloc[i-5:i+1].max()
            low_range_6 = current_data['low'].iloc[i-5:i+1].min()
            volume_t = current_data['volume'].iloc[i]
            amount_t = current_data['amount'].iloc[i]
            
            price_compression = ((high_range_3 - low_range_3) / (high_range_6 - low_range_6 + 1e-8) * 
                               volume_t / (amount_t + 1e-8))
        else:
            price_compression = 0
        
        # Fractal Flow Asymmetry
        # Quantum Flow Imbalance
        if i >= 7:
            amount_data = current_data['amount'].iloc[i-6:i+1]
            close_data = current_data['close'].iloc[i-6:i+1]
            open_data = current_data['open'].iloc[i-6:i+1]
            
            positive_amount = 0
            negative_amount = 0
            total_amount = 0
            
            for j in range(len(amount_data)):
                if close_data.iloc[j] > open_data.iloc[j]:
                    positive_amount += amount_data.iloc[j]
                elif close_data.iloc[j] < open_data.iloc[j]:
                    negative_amount += amount_data.iloc[j]
                total_amount += amount_data.iloc[j]
            
            flow_imbalance = (positive_amount - negative_amount) / (total_amount + 1e-8)
        else:
            flow_imbalance = 0
        
        # Entropic Volume Pressure
        if i >= 1:
            volume_t = current_data['volume'].iloc[i]
            volume_t_minus_1 = current_data['volume'].iloc[i-1]
            close_t = current_data['close'].iloc[i]
            open_t = current_data['open'].iloc[i]
            close_t_minus_1 = current_data['close'].iloc[i-1]
            
            volume_pressure = (volume_t / (volume_t_minus_1 + 1e-8) * 
                             np.sign(close_t - open_t) * 
                             abs(open_t - close_t_minus_1) / (close_t_minus_1 + 1e-8))
        else:
            volume_pressure = 0
        
        # Quantum Microstructure Reversal
        # High-frequency reversal
        if i >= 1:
            close_t = current_data['close'].iloc[i]
            open_t = current_data['open'].iloc[i]
            high_t = current_data['high'].iloc[i]
            low_t = current_data['low'].iloc[i]
            volume_t = current_data['volume'].iloc[i]
            volume_t_minus_1 = current_data['volume'].iloc[i-1]
            close_t_minus_1 = current_data['close'].iloc[i-1]
            
            high_freq_reversal = ((close_t - open_t) / (high_t - low_t + 1e-8) * 
                                volume_t / (volume_t_minus_1 + 1e-8) * 
                                abs(open_t - close_t_minus_1) / (close_t_minus_1 + 1e-8) * 
                                np.sign(close_t - open_t) * -1)
        else:
            high_freq_reversal = 0
        
        # Quantum Closing Momentum
        if i >= 2:
            close_t = current_data['close'].iloc[i]
            open_t = current_data['open'].iloc[i]
            high_range_2 = current_data['high'].iloc[i-1:i+1].max()
            low_range_2 = current_data['low'].iloc[i-1:i+1].min()
            volume_t = current_data['volume'].iloc[i]
            volume_t_minus_2 = current_data['volume'].iloc[i-2]
            
            closing_momentum = ((close_t - open_t) / (high_range_2 - low_range_2 + 1e-8) * 
                              np.sign(volume_t - volume_t_minus_2))
        else:
            closing_momentum = 0
        
        # Entropic Volume Reversal
        # Quantum Volume Asymmetry
        if i >= 7:
            volume_data = current_data['volume'].iloc[i-6:i+1]
            close_data = current_data['close'].iloc[i-6:i+1]
            open_data = current_data['open'].iloc[i-6:i+1]
            
            up_volume = 0
            down_volume = 0
            
            for j in range(len(volume_data)):
                if close_data.iloc[j] > open_data.iloc[j]:
                    up_volume += volume_data.iloc[j]
                elif close_data.iloc[j] < open_data.iloc[j]:
                    down_volume += volume_data.iloc[j]
            
            volume_asymmetry = (up_volume / (down_volume + 1e-8) * 
                              np.sign(current_data['volume'].iloc[i] - current_data['volume'].iloc[i-3]))
        else:
            volume_asymmetry = 0
        
        # Quantum Volume Shock
        if i >= 2:
            volume_t = current_data['volume'].iloc[i]
            volume_t_minus_2 = current_data['volume'].iloc[i-2]
            
            if volume_t > 2.0 * volume_t_minus_2:
                volume_shock = -1
            elif volume_t < 0.5 * volume_t_minus_2:
                volume_shock = 1
            else:
                volume_shock = 0
        else:
            volume_shock = 0
        
        # Fractal Integration Framework
        # Entropic-Momentum Core
        entropic_momentum_core = bidirectional_momentum * flow_imbalance * volume_pressure
        
        # Microstructure-Reversal Alignment
        microstructure_alignment = ((1 - price_compression) * volume_asymmetry * high_freq_reversal)
        
        # Quantum Position Signal
        if i >= 5:
            close_t = current_data['close'].iloc[i]
            low_range_5 = current_data['low'].iloc[i-4:i+1].min()
            high_range_5 = current_data['high'].iloc[i-4:i+1].max()
            open_t = current_data['open'].iloc[i]
            
            position_signal = ((close_t - low_range_5) / (high_range_5 - low_range_5 + 1e-8) * 
                             flow_imbalance * np.sign(close_t - open_t))
        else:
            position_signal = 0
        
        # Base Fractal Signal
        base_fractal_signal = entropic_momentum_core * microstructure_alignment * position_signal
        
        # Quantum Fractal Regime Classification
        regime_multiplier = 1.0
        
        if high_freq_reversal > 0.6 and closing_momentum > 0:
            regime_multiplier = 1.4
        elif 0.3 <= high_freq_reversal <= 0.6 and volume_asymmetry > 0:
            regime_multiplier = 1.0
        elif high_freq_reversal < 0.3 and bidirectional_momentum < 0:
            regime_multiplier = 0.6
        
        # Final Quantum Fractal Alpha
        final_alpha = base_fractal_signal * regime_multiplier * volume_shock
        
        result.iloc[i] = final_alpha
    
    # Fill early NaN values with 0
    result = result.fillna(0)
    
    return result
