import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize all required columns
    data['micro_fractal_flow'] = 0.0
    data['meso_fractal_flow'] = 0.0
    data['macro_fractal_flow'] = 0.0
    data['fractal_flow_transmission'] = 0.0
    data['volume_flow_fractal'] = 0.0
    data['volatility_regime_volume_divergence'] = 0.0
    data['volume_path_regime_persistence'] = 0
    data['volatility_adjusted_gap_momentum'] = 0.0
    data['gap_flow_signal'] = 0.0
    data['micro_volatility_expansion'] = 0.0
    data['volatility_transmission_score'] = 0.0
    data['volatility_regime_divergence_signal'] = 0.0
    data['price_flow_divergence'] = 0.0
    data['flow_divergence_transmission'] = 0.0
    data['volume_volatility_divergence_strength'] = 0.0
    data['price_fractal_consistency'] = 0.0
    data['fractal_quality_score'] = 0.0
    data['range_turning_point_momentum'] = 0.0
    
    # Calculate rolling windows
    for i in range(len(data)):
        if i < 8:  # Need at least 8 days for calculations
            continue
            
        # Multi-Scale Fractal Dynamics
        # Micro-Fractal Flow
        micro_ff = ((data['close'].iloc[i] - data['open'].iloc[i]) / (data['volume'].iloc[i] + 0.001)) * \
                  ((data['high'].iloc[i] - data['close'].iloc[i]) / (data['close'].iloc[i] - data['low'].iloc[i] + 0.001))
        data.loc[data.index[i], 'micro_fractal_flow'] = micro_ff
        
        # Meso-Fractal Flow
        high_2_0 = data['high'].iloc[i-2:i+1].max()
        low_2_0 = data['low'].iloc[i-2:i+1].min()
        meso_ff = ((data['close'].iloc[i] - data['close'].iloc[i-3]) / (data['volume'].iloc[i] + 0.001)) * \
                 ((high_2_0 - data['close'].iloc[i]) / (data['close'].iloc[i] - low_2_0 + 0.001))
        data.loc[data.index[i], 'meso_fractal_flow'] = meso_ff
        
        # Macro-Fractal Flow
        high_5_0 = data['high'].iloc[i-5:i+1].max()
        low_5_0 = data['low'].iloc[i-5:i+1].min()
        macro_ff = ((data['close'].iloc[i] - data['close'].iloc[i-5]) / (data['volume'].iloc[i] + 0.001)) * \
                  ((high_5_0 - data['close'].iloc[i]) / (data['close'].iloc[i] - low_5_0 + 0.001))
        data.loc[data.index[i], 'macro_fractal_flow'] = macro_ff
        
        # Fractal Flow Transmission
        fft = micro_ff * meso_ff * macro_ff * np.sign(meso_ff - macro_ff)
        data.loc[data.index[i], 'fractal_flow_transmission'] = fft
        
        # Volume-Volatility Synchronization
        # Volume Flow Fractal
        high_8_0 = data['high'].iloc[i-8:i+1].max()
        low_8_0 = data['low'].iloc[i-8:i+1].min()
        vff = data['volume'].iloc[i] * ((data['close'].iloc[i] - data['open'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i] + 0.001)) * \
              (data['volume'].iloc[i] / (data['volume'].iloc[i-8] + 0.001)) * \
              ((data['close'].iloc[i] - data['close'].iloc[i-8]) / (high_8_0 - low_8_0 + 0.001))
        data.loc[data.index[i], 'volume_flow_fractal'] = vff
        
        # Volatility Regime Volume Divergence
        vrv_div = ((data['volume'].iloc[i] - data['volume'].iloc[i-2]) / (data['volume'].iloc[i-2] + 0.001)) - \
                 ((data['high'].iloc[i] - data['low'].iloc[i]) / (data['high'].iloc[i-2] - data['low'].iloc[i-2] + 0.001))
        data.loc[data.index[i], 'volatility_regime_volume_divergence'] = vrv_div
        
        # Volume-Path Regime Persistence
        vol_inc_count = 0
        vol_dec_count = 0
        for j in range(1, 6):
            if i-j >= 0:
                if data['volume'].iloc[i-j] > data['volume'].iloc[i-j-1]:
                    vol_inc_count += 1
                elif data['volume'].iloc[i-j] < data['volume'].iloc[i-j-1]:
                    vol_dec_count += 1
        
        vol_sign_count = 0
        for j in range(3):
            if i-2-j >= 0:
                vol_ratio = (data['high'].iloc[i-2-j] - data['low'].iloc[i-2-j]) / (data['high'].iloc[i-7-j] - data['low'].iloc[i-7-j] + 0.001) - 1
                if j > 0:
                    prev_ratio = (data['high'].iloc[i-1-j] - data['low'].iloc[i-1-j]) / (data['high'].iloc[i-6-j] - data['low'].iloc[i-6-j] + 0.001) - 1
                    if np.sign(vol_ratio) == np.sign(prev_ratio):
                        vol_sign_count += 1
        
        vprp = (vol_inc_count - vol_dec_count) * vol_sign_count
        data.loc[data.index[i], 'volume_path_regime_persistence'] = vprp
        
        # Gap Transmission Dynamics
        # Volatility-Adjusted Gap Momentum
        vgm = ((data['open'].iloc[i] - data['close'].iloc[i-1]) / (data['close'].iloc[i-1] + 0.001)) * \
              ((data['volume'].iloc[i] - data['volume'].iloc[i-1]) / (data['volume'].iloc[i] + data['volume'].iloc[i-1] + 0.001)) * \
              np.sign(data['close'].iloc[i] - data['close'].iloc[i-1]) * \
              ((data['close'].iloc[i] - data['open'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i] + 0.001))
        data.loc[data.index[i], 'volatility_adjusted_gap_momentum'] = vgm
        
        # Gap Flow Signal
        gfs = vgm * np.sign(data['close'].iloc[i] - data['close'].iloc[i-1])
        data.loc[data.index[i], 'gap_flow_signal'] = gfs
        
        # Fractal Volatility Transmission
        # Micro Volatility Expansion
        mve = ((data['high'].iloc[i] - data['low'].iloc[i]) / (data['high'].iloc[i-3] - data['low'].iloc[i-3] + 0.001) - 1) * \
              ((data['close'].iloc[i] - data['close'].iloc[i-1]) / (data['high'].iloc[i] - data['low'].iloc[i] + 0.001))
        data.loc[data.index[i], 'micro_volatility_expansion'] = mve
        
        # Volatility Transmission Score
        vts = np.sign(mve) * np.sqrt(data['high'].iloc[i] - data['low'].iloc[i])
        data.loc[data.index[i], 'volatility_transmission_score'] = vts
        
        # Volatility Regime Classification (simplified)
        vol_regime = 1 if (data['high'].iloc[i] - data['low'].iloc[i]) > (data['high'].iloc[i-5] - data['low'].iloc[i-5]) else -1
        
        # Volatility Regime Divergence Signal
        vrds = np.sign(fft) * np.sign(vrv_div) * \
               ((data['close'].iloc[i] - data['close'].iloc[i-1]) / (data['high'].iloc[i] - data['low'].iloc[i] + 0.001)) * \
               vol_regime
        data.loc[data.index[i], 'volatility_regime_divergence_signal'] = vrds
        
        # Flow Divergence Transmission
        # Price-Flow Divergence
        pfd = ((data['close'].iloc[i] - data['close'].iloc[i-2]) / (data['close'].iloc[i-2] + 0.001)) - \
              (((data['high'].iloc[i] - data['close'].iloc[i]) - (data['close'].iloc[i] - data['low'].iloc[i])) / (data['high'].iloc[i-2] - data['low'].iloc[i-2] + 0.001))
        data.loc[data.index[i], 'price_flow_divergence'] = pfd
        
        # Flow Divergence Transmission
        fdt = np.sign(pfd) * ((data['close'].iloc[i] - data['close'].iloc[i-1]) / (data['high'].iloc[i] - data['low'].iloc[i] + 0.001))
        data.loc[data.index[i], 'flow_divergence_transmission'] = fdt
        
        # Volume-Volatility Divergence Strength
        if data['close'].iloc[i] > data['close'].iloc[i-1] and data['volume'].iloc[i] < data['volume'].iloc[i-1]:
            vvds_base = 1
        elif data['close'].iloc[i] < data['close'].iloc[i-1] and data['volume'].iloc[i] > data['volume'].iloc[i-1]:
            vvds_base = -1
        else:
            vvds_base = 0
            
        vvds = vvds_base * abs(data['close'].iloc[i] - data['close'].iloc[i-1]) / (data['high'].iloc[i] - data['low'].iloc[i] + 0.001) * \
               ((data['volume'].iloc[i] - data['volume'].iloc[i-5]) / (data['volume'].iloc[i-5] + 0.001))
        data.loc[data.index[i], 'volume_volatility_divergence_strength'] = vvds
        
        # Fractal Synchronization Quality
        # Price Fractal Consistency
        high_2_0 = data['high'].iloc[i-2:i+1].max()
        low_2_0 = data['low'].iloc[i-2:i+1].min()
        high_5_0 = data['high'].iloc[i-5:i+1].max()
        low_5_0 = data['low'].iloc[i-5:i+1].min()
        
        pfc = ((data['high'].iloc[i] - data['low'].iloc[i]) / (high_2_0 - low_2_0 + 0.001)) * \
              ((high_2_0 - low_2_0) / (high_5_0 - low_5_0 + 0.001))
        data.loc[data.index[i], 'price_fractal_consistency'] = pfc
        
        # Fractal Quality Score (simplified)
        fft_consecutive = 0
        for j in range(3):
            if i-j-1 >= 0 and np.sign(data['fractal_flow_transmission'].iloc[i-j]) == np.sign(data['fractal_flow_transmission'].iloc[i-j-1]):
                fft_consecutive += 1
        
        fqs = pfc * fft_consecutive
        data.loc[data.index[i], 'fractal_quality_score'] = fqs
        
        # Range-Turning Point Momentum
        range_inc_count = 0
        range_dec_count = 0
        for j in range(3):
            if i-j-1 >= 0:
                if (data['high'].iloc[i-j] - data['low'].iloc[i-j]) > (data['high'].iloc[i-j-1] - data['low'].iloc[i-j-1]):
                    range_inc_count += 1
                elif (data['high'].iloc[i-j] - data['low'].iloc[i-j]) < (data['high'].iloc[i-j-1] - data['low'].iloc[i-j-1]):
                    range_dec_count += 1
        
        turning_point_density = range_inc_count + range_dec_count
        rtpm = (range_inc_count - range_dec_count) * (1 - turning_point_density / 5)
        data.loc[data.index[i], 'range_turning_point_momentum'] = rtpm
    
    # Composite Alpha Construction
    alpha_values = []
    
    for i in range(len(data)):
        if i < 8:
            alpha_values.append(0.0)
            continue
            
        # Core Transmission Factors
        fractal_flow_core = data['fractal_flow_transmission'].iloc[i] * data['volume_path_regime_persistence'].iloc[i]
        gap_transmission_core = data['volatility_adjusted_gap_momentum'].iloc[i] * data['volatility_transmission_score'].iloc[i]
        divergence_transmission_core = data['flow_divergence_transmission'].iloc[i] * data['fractal_quality_score'].iloc[i] * data['volatility_regime_divergence_signal'].iloc[i]
        
        # Synchronization Enhancement
        volume_volatility_sync = data['volume_flow_fractal'].iloc[i] * data['volatility_regime_volume_divergence'].iloc[i] * data['range_turning_point_momentum'].iloc[i]
        divergence_multiplier = data['volume_volatility_divergence_strength'].iloc[i] * data['volatility_regime_divergence_signal'].iloc[i]
        
        # Final Alpha Calculation
        base_transmission_alpha = fractal_flow_core * gap_transmission_core * divergence_transmission_core
        enhanced_sync_alpha = base_transmission_alpha * volume_volatility_sync * (1 + abs(divergence_multiplier))
        
        # Volatility Regime Classification (simplified)
        vol_regime = 1 if (data['high'].iloc[i] - data['low'].iloc[i]) > (data['high'].iloc[i-5] - data['low'].iloc[i-5]) else -1
        
        final_alpha = enhanced_sync_alpha * np.sign(data['fractal_flow_transmission'].iloc[i]) * vol_regime
        
        alpha_values.append(final_alpha)
    
    # Create output series
    alpha_series = pd.Series(alpha_values, index=data.index)
    
    # Handle any potential infinite values
    alpha_series = alpha_series.replace([np.inf, -np.inf], 0)
    
    return alpha_series
