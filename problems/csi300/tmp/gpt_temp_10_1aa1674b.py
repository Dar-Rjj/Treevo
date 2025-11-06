import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate required rolling windows
    for i in range(len(data)):
        if i < 16:  # Need at least 16 days for calculations
            alpha.iloc[i] = 0
            continue
            
        # Extract current and historical data
        current = data.iloc[i]
        prev_1 = data.iloc[i-1] if i >= 1 else None
        prev_3 = data.iloc[i-3] if i >= 3 else None
        prev_6 = data.iloc[i-6] if i >= 6 else None
        prev_8 = data.iloc[i-8] if i >= 8 else None
        prev_16 = data.iloc[i-16] if i >= 16 else None
        
        # Multi-Scale Fracture Momentum
        # Micro Fracture Momentum
        micro_frac_current = (current['close'] - current['open']) / (current['high'] - current['low'] + 1e-8)
        micro_frac_prev = (prev_1['close'] - prev_1['open']) / (prev_1['high'] - prev_1['low'] + 1e-8)
        micro_frac_mom = micro_frac_current - micro_frac_prev
        
        # Meso Fracture Momentum
        high_3_0 = data['high'].iloc[i-3:i+1].max()
        low_3_0 = data['low'].iloc[i-3:i+1].min()
        high_6_3 = data['high'].iloc[i-6:i-2].max()
        low_6_3 = data['low'].iloc[i-6:i-2].min()
        
        meso_frac_current = (current['close'] - prev_3['close']) / (high_3_0 - low_3_0 + 1e-8)
        meso_frac_prev = (prev_3['close'] - prev_6['close']) / (high_6_3 - low_6_3 + 1e-8)
        meso_frac_mom = meso_frac_current - meso_frac_prev
        
        # Macro Fracture Momentum
        high_8_0 = data['high'].iloc[i-8:i+1].max()
        low_8_0 = data['low'].iloc[i-8:i+1].min()
        high_16_8 = data['high'].iloc[i-16:i-7].max()
        low_16_8 = data['low'].iloc[i-16:i-7].min()
        
        macro_frac_current = (current['close'] - prev_8['close']) / (high_8_0 - low_8_0 + 1e-8)
        macro_frac_prev = (prev_8['close'] - prev_16['close']) / (high_16_8 - low_16_8 + 1e-8)
        macro_frac_mom = macro_frac_current - macro_frac_prev
        
        # Fracture Momentum Cascade
        frac_mom_cascade = micro_frac_mom * meso_frac_mom * macro_frac_mom
        
        # Multi-Scale Efficiency
        micro_eff = micro_frac_current
        meso_eff = meso_frac_current
        macro_eff = macro_frac_current
        eff_cascade = micro_eff * meso_eff * macro_eff
        
        # Fracture-Efficiency Convergence
        micro_conv = np.sign(micro_frac_mom) * np.sign(micro_eff)
        meso_conv = np.sign(meso_frac_mom) * np.sign(meso_eff)
        macro_conv = np.sign(macro_frac_mom) * np.sign(macro_eff)
        conv_strength = sum([1 if x > 0 else 0 for x in [micro_conv, meso_conv, macro_conv]])
        
        # Volume-Pressure Enhancement
        # Volume-Weighted Fracture Momentum
        current_vol_mom = (current['close'] - current['open']) * current['volume'] / (current['high'] - current['low'] + 1e-8)
        prev_vol_mom = (prev_1['close'] - prev_1['open']) * prev_1['volume'] / (prev_1['high'] - prev_1['low'] + 1e-8)
        vol_mom_frac = current_vol_mom - prev_vol_mom
        
        # Volume Asymmetry Dynamics
        morning_vol_pressure = current['volume'] * micro_frac_current * (current['high'] - current['open'])
        afternoon_vol_pressure = current['volume'] * micro_frac_current * (current['open'] - current['low'])
        vol_asymmetry_signal = morning_vol_pressure - afternoon_vol_pressure
        
        # Pressure Differential Enhancement
        pressure_diff = (current['high'] - current['open']) * (current['close'] - current['low']) - (current['open'] - current['low']) * (current['high'] - current['close'])
        enhanced_vol_pressure = vol_asymmetry_signal * pressure_diff
        
        # Microstructure Alignment
        # Trade Size Dynamics
        avg_trade_size = current['amount'] / (current['volume'] + 1e-8)
        prev_avg_trade_size = prev_1['amount'] / (prev_1['volume'] + 1e-8)
        trade_size_mom = avg_trade_size / (prev_avg_trade_size + 1e-8)
        trade_size_pressure = avg_trade_size * micro_frac_current
        
        # Microstructure Asymmetry
        large_trade_pressure = current['amount'] * micro_frac_current * (current['high'] - current['open'])
        small_trade_pressure = current['volume'] * ((current['close'] - current['low']) / (current['high'] - current['low'] + 1e-8)) * (current['open'] - current['low'])
        trade_size_divergence = large_trade_pressure - small_trade_pressure
        
        # Volume-Amount Alignment
        vol_alignment = np.sign(current['volume'] / (prev_1['volume'] + 1e-8)) * np.sign(vol_asymmetry_signal)
        amount_alignment = np.sign(current['amount'] / (prev_1['amount'] + 1e-8)) * np.sign(trade_size_divergence)
        microstructure_sync = vol_alignment * amount_alignment
        
        # Persistence Framework
        # Fracture Momentum Persistence
        micro_frac_signs = []
        for j in range(1, 4):
            if i-j >= 0:
                micro_frac_prev_j = (data.iloc[i-j]['close'] - data.iloc[i-j]['open']) / (data.iloc[i-j]['high'] - data.iloc[i-j]['low'] + 1e-8)
                micro_frac_prev_j_1 = (data.iloc[i-j-1]['close'] - data.iloc[i-j-1]['open']) / (data.iloc[i-j-1]['high'] - data.iloc[i-j-1]['low'] + 1e-8)
                micro_frac_mom_j = micro_frac_prev_j - micro_frac_prev_j_1
                micro_frac_signs.append(np.sign(micro_frac_mom_j))
        
        short_term_frac_persistence = sum([1 for x in micro_frac_signs if x > 0]) - sum([1 for x in micro_frac_signs if x < 0])
        
        micro_frac_signs_medium = []
        for j in range(1, 7):
            if i-j >= 0 and i-j-1 >= 0:
                micro_frac_prev_j = (data.iloc[i-j]['close'] - data.iloc[i-j]['open']) / (data.iloc[i-j]['high'] - data.iloc[i-j]['low'] + 1e-8)
                micro_frac_prev_j_1 = (data.iloc[i-j-1]['close'] - data.iloc[i-j-1]['open']) / (data.iloc[i-j-1]['high'] - data.iloc[i-j-1]['low'] + 1e-8)
                micro_frac_mom_j = micro_frac_prev_j - micro_frac_prev_j_1
                micro_frac_signs_medium.append(np.sign(micro_frac_mom_j))
        
        medium_term_frac_persistence = sum([1 for x in micro_frac_signs_medium if x > 0]) - sum([1 for x in micro_frac_signs_medium if x < 0])
        frac_persistence_ratio = short_term_frac_persistence / (medium_term_frac_persistence + 1e-8)
        
        # Efficiency Persistence
        micro_eff_persistence = 0
        for j in range(1, 3):
            if i-j >= 0:
                micro_eff_j = (data.iloc[i-j]['close'] - data.iloc[i-j]['open']) / (data.iloc[i-j]['high'] - data.iloc[i-j]['low'] + 1e-8)
                if np.sign(micro_eff_j) == np.sign(micro_eff):
                    micro_eff_persistence += 1
        
        meso_eff_persistence = 0
        for j in range(1, 4):
            if i-j >= 3:
                high_j = data['high'].iloc[i-j-3:i-j+1].max()
                low_j = data['low'].iloc[i-j-3:i-j+1].min()
                meso_eff_j = (data.iloc[i-j]['close'] - data.iloc[i-j-3]['close']) / (high_j - low_j + 1e-8)
                if np.sign(meso_eff_j) == np.sign(meso_eff):
                    meso_eff_persistence += 1
        
        macro_eff_persistence = 0
        for j in range(1, 6):
            if i-j >= 8:
                high_j = data['high'].iloc[i-j-8:i-j+1].max()
                low_j = data['low'].iloc[i-j-8:i-j+1].min()
                macro_eff_j = (data.iloc[i-j]['close'] - data.iloc[i-j-8]['close']) / (high_j - low_j + 1e-8)
                if np.sign(macro_eff_j) == np.sign(macro_eff):
                    macro_eff_persistence += 1
        
        combined_eff_persistence = micro_eff_persistence * meso_eff_persistence * macro_eff_persistence
        
        # Volume-Pressure Persistence
        vol_asymmetry_persistence = 0
        for j in range(1, 3):
            if i-j >= 0:
                vol_asymmetry_j = data.iloc[i-j]['volume'] * ((data.iloc[i-j]['close'] - data.iloc[i-j]['open']) / (data.iloc[i-j]['high'] - data.iloc[i-j]['low'] + 1e-8)) * (data.iloc[i-j]['high'] - data.iloc[i-j]['open']) - data.iloc[i-j]['volume'] * ((data.iloc[i-j]['close'] - data.iloc[i-j]['open']) / (data.iloc[i-j]['high'] - data.iloc[i-j]['low'] + 1e-8)) * (data.iloc[i-j]['open'] - data.iloc[i-j]['low'])
                if np.sign(vol_asymmetry_j) == np.sign(vol_asymmetry_signal):
                    vol_asymmetry_persistence += 1
        
        vol_timing_persistence = 0
        for j in range(1, 3):
            if i-j >= 1:
                vol_mom_frac_j = (data.iloc[i-j]['close'] - data.iloc[i-j]['open']) * data.iloc[i-j]['volume'] / (data.iloc[i-j]['high'] - data.iloc[i-j]['low'] + 1e-8) - (data.iloc[i-j-1]['close'] - data.iloc[i-j-1]['open']) * data.iloc[i-j-1]['volume'] / (data.iloc[i-j-1]['high'] - data.iloc[i-j-1]['low'] + 1e-8)
                if np.sign(vol_mom_frac_j) == np.sign(vol_mom_frac):
                    vol_timing_persistence += 1
        
        combined_vol_persistence = vol_asymmetry_persistence * vol_timing_persistence
        
        # Volatility Transmission
        # Multi-Scale Volatility
        micro_vol = (current['high'] - current['low']) / (current['open'] + 1e-8)
        meso_vol = (high_3_0 - low_3_0) / (prev_3['close'] + 1e-8)
        macro_vol = (high_8_0 - low_8_0) / (prev_8['close'] + 1e-8)
        
        # Volatility Transmission Signal
        volatility_cascade = 0
        for j in range(3):
            if i-j >= 8:
                micro_vol_j = (data.iloc[i-j]['high'] - data.iloc[i-j]['low']) / (data.iloc[i-j]['open'] + 1e-8)
                high_3_j = data['high'].iloc[i-j-3:i-j+1].max()
                low_3_j = data['low'].iloc[i-j-3:i-j+1].min()
                meso_vol_j = (high_3_j - low_3_j) / (data.iloc[i-j-3]['close'] + 1e-8)
                high_8_j = data['high'].iloc[i-j-8:i-j+1].max()
                low_8_j = data['low'].iloc[i-j-8:i-j+1].min()
                macro_vol_j = (high_8_j - low_8_j) / (data.iloc[i-j-8]['close'] + 1e-8)
                
                if micro_vol_j > meso_vol_j and meso_vol_j > macro_vol_j:
                    volatility_cascade += 1
        
        volatility_adjusted_frac = frac_mom_cascade / (micro_vol + 1e-8)
        
        # Hierarchical Alpha Synthesis
        # Core Components Integration
        fracture_eff_core = frac_mom_cascade * eff_cascade * conv_strength
        vol_pressure_core = enhanced_vol_pressure * vol_mom_frac * microstructure_sync
        persistence_core = frac_persistence_ratio * combined_eff_persistence * combined_vol_persistence
        
        # Volatility-Enhanced Framework
        vol_enhanced_frac = fracture_eff_core / (meso_vol + 1e-8)
        vol_enhanced_vol = vol_pressure_core / (micro_vol + 1e-8)
        vol_transmission_integration = vol_enhanced_frac * vol_enhanced_vol * volatility_cascade
        
        # Persistence-Refined Components
        persistence_enhanced_frac = vol_transmission_integration * persistence_core
        microstructure_pressure_quality = trade_size_divergence * trade_size_pressure
        enhanced_microstructure = microstructure_pressure_quality * microstructure_sync
        
        # Final Alpha Factor
        synchronization_framework = persistence_enhanced_frac * enhanced_microstructure * pressure_diff
        final_alpha = synchronization_framework * (current['close'] - current['open'])
        
        alpha.iloc[i] = final_alpha
    
    return alpha
