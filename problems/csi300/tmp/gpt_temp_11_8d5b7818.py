import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required columns with proper shifting
    df['prev_close'] = df['close'].shift(1)
    df['prev_open'] = df['open'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    df['prev_volume2'] = df['volume'].shift(2)
    
    # Rolling windows for historical calculations
    df['range_5d'] = df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()
    df['range_20d'] = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
    df['volume_5d_avg'] = df['volume'].rolling(window=5).mean()
    df['high_5d_avg'] = df['high'].rolling(window=5).mean()
    df['low_5d_avg'] = df['low'].rolling(window=5).mean()
    
    for i in range(len(df)):
        if i < 20:  # Skip first 20 days for sufficient data
            result.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        
        # Asymmetric Volatility Dynamics
        high_low_range = current['high'] - current['low']
        if high_low_range > 0:
            upward_fracture = (current['high'] - current['close']) / high_low_range
            downward_fracture = (current['close'] - current['low']) / high_low_range
            volatility_asymmetry = upward_fracture / downward_fracture - 1 if downward_fracture > 0 else 0
        else:
            upward_fracture = downward_fracture = volatility_asymmetry = 0
        
        # Multi-Timeframe Volatility Fracture
        if i >= 5 and df.iloc[i-5]['high'] - df.iloc[i-5]['low'] > 0:
            short_term_compression = high_low_range / (df.iloc[i-5]['high'] - df.iloc[i-5]['low']) - 1
        else:
            short_term_compression = 0
            
        if df.iloc[i-20]['high'] - df.iloc[i-20]['low'] > 0:
            medium_term_memory = abs(current['close'] - df.iloc[i-20]['close']) / (df.iloc[i-20]['high'] - df.iloc[i-20]['low'])
        else:
            medium_term_memory = 0
            
        volatility_fracture_signal = volatility_asymmetry * short_term_compression
        
        # Session Boundary Volatility
        if i > 0 and abs(current['open'] - df.iloc[i-1]['close']) > 0:
            opening_gap_absorption = (high_low_range / abs(current['open'] - df.iloc[i-1]['close'])) * np.sign(current['close'] - current['open'])
        else:
            opening_gap_absorption = 0
            
        if high_low_range > 0:
            intraday_efficiency = abs(current['close'] - current['open']) / high_low_range
        else:
            intraday_efficiency = 0
            
        boundary_volatility_signal = opening_gap_absorption * intraday_efficiency
        
        # Bidirectional Momentum Decay
        if i > 0 and df.iloc[i-1]['close'] > 0:
            opening_flow_momentum = (current['open'] - df.iloc[i-1]['close']) / df.iloc[i-1]['close']
            closing_flow_momentum = (current['close'] - current['open']) / current['open'] if current['open'] > 0 else 0
            flow_divergence = abs(opening_flow_momentum - closing_flow_momentum)
        else:
            opening_flow_momentum = closing_flow_momentum = flow_divergence = 0
        
        # Momentum Decay Patterns
        if i > 0 and abs(current['close'] - (current['high'] + current['low'])/2) > 0:
            price_level_fracture = (current['close'] - df.iloc[i-1]['close']) / abs(current['close'] - (current['high'] + current['low'])/2)
        else:
            price_level_fracture = 0
            
        if i > 1 and df.iloc[i-1]['volume'] > 0 and abs(df.iloc[i-1]['close'] - df.iloc[i-2]['close']) > 0:
            volume_weighted_decay = (current['volume'] / df.iloc[i-1]['volume']) * ((current['close'] - df.iloc[i-1]['close']) / abs(df.iloc[i-1]['close'] - df.iloc[i-2]['close']))
        else:
            volume_weighted_decay = 0
            
        momentum_decay_signal = price_level_fracture * volume_weighted_decay
        
        # Microstructure Momentum
        if high_low_range > 0:
            session_dominance = (max(current['high'] - current['open'], current['open'] - current['low']) / high_low_range) * np.sign(current['close'] - current['open'])
        else:
            session_dominance = 0
            
        if i > 0 and high_low_range > 0 and (df.iloc[i-1]['high'] - df.iloc[i-1]['low']) > 0:
            closing_persistence = ((current['close'] - current['low']) / high_low_range) * ((df.iloc[i-1]['close'] - df.iloc[i-1]['low']) / (df.iloc[i-1]['high'] - df.iloc[i-1]['low']))
        else:
            closing_persistence = 0
            
        microstructure_signal = session_dominance * closing_persistence
        
        # Volume-Pressure Fractal Alignment
        if i >= 2:
            volume_range_fracture = max(df.iloc[i-2]['volume'], df.iloc[i-1]['volume'], current['volume']) - min(df.iloc[i-2]['volume'], df.iloc[i-1]['volume'], current['volume'])
            volume_path_intensity = abs(current['volume'] - df.iloc[i-1]['volume']) + abs(df.iloc[i-1]['volume'] - df.iloc[i-2]['volume'])
            volume_fractal_signal = volume_range_fracture * volume_path_intensity
        else:
            volume_fractal_signal = 0
        
        # Pressure Fracture Components
        if high_low_range > 0:
            directional_pressure = current['volume'] * (current['close'] - current['open']) / high_low_range
        else:
            directional_pressure = 0
            
        if current['volume_5d_avg'] > 0:
            pressure_clustering = current['volume'] / current['volume_5d_avg']
        else:
            pressure_clustering = 0
            
        pressure_fracture_signal = directional_pressure * pressure_clustering
        
        # Fractal Alignment
        volume_pressure_correlation = volume_fractal_signal * pressure_fracture_signal
        fractal_divergence = volume_fractal_signal - pressure_fracture_signal
        alignment_signal = volume_pressure_correlation * fractal_divergence
        
        # Regime-Adaptive Signal Synthesis
        high_volatility_condition = 1 if high_low_range > (current['high_5d_avg'] - current['low_5d_avg']) else 0
        low_volatility_condition = 1 if high_low_range < (current['high_5d_avg'] - current['low_5d_avg']) else 0
        regime_signal = high_volatility_condition * 0.6 + low_volatility_condition * 0.4
        
        decay_regime = 1 if flow_divergence > 0.1 and momentum_decay_signal < 0 else 0
        persistence_regime = 1 if flow_divergence < 0.1 and momentum_decay_signal > 0 else 0
        regime_adaptation = decay_regime * 0.5 + persistence_regime * 0.5
        
        # Microstructure Validation
        multi_scale_consistency = volatility_fracture_signal * momentum_decay_signal * alignment_signal
        volume_pressure_sync = pressure_fracture_signal * microstructure_signal
        validation_signal = multi_scale_consistency * volume_pressure_sync
        
        # Composite Fractal Alpha
        base_fractal = volatility_fracture_signal * momentum_decay_signal
        enhanced_fractal = base_fractal * alignment_signal
        regime_enhanced = enhanced_fractal * regime_signal
        
        flow_refinement = flow_divergence * microstructure_signal
        pressure_refinement = pressure_fracture_signal * volume_fractal_signal
        refined_signal = flow_refinement * pressure_refinement
        
        # Final Alpha Generation
        integrated_factor = regime_enhanced * refined_signal * validation_signal
        result.iloc[i] = integrated_factor
    
    # Clean up intermediate columns
    df.drop(['prev_close', 'prev_open', 'prev_high', 'prev_low', 'prev_volume', 'prev_volume2', 
             'range_5d', 'range_20d', 'volume_5d_avg', 'high_5d_avg', 'low_5d_avg'], axis=1, inplace=True, errors='ignore')
    
    return result
