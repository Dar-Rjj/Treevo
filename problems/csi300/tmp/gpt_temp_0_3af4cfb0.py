import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required components with proper shifting to avoid lookahead
    for i in range(len(df)):
        if i < 20:  # Need sufficient history for calculations
            result.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        # Historical data - only use past data
        hist_1 = df.iloc[i-1] if i >= 1 else None
        hist_2 = df.iloc[i-2] if i >= 2 else None
        hist_4 = df.iloc[i-4] if i >= 4 else None
        hist_9 = df.iloc[i-9] if i >= 9 else None
        
        # Calculate rolling windows using only past data
        vol_window = df.iloc[max(0, i-19):i+1]  # t-19 to t
        volume_window = df.iloc[max(0, i-5):i]  # t-5 to t-1
        
        # Asymmetric Volatility-Momentum Dynamics
        # Directional momentum efficiency
        if current['high'] != current['low']:
            if current['close'] > current['open']:
                upside_momentum_persistence = (current['close'] - current['open']) / (current['high'] - current['low'])
            else:
                upside_momentum_persistence = 0
                
            if current['close'] < current['open']:
                downside_momentum_clustering = (current['open'] - current['close']) / (current['high'] - current['low'])
            else:
                downside_momentum_clustering = 0
        else:
            upside_momentum_persistence = 0
            downside_momentum_clustering = 0
            
        # Avoid division by zero
        if downside_momentum_clustering != 0:
            momentum_asymmetry_ratio = upside_momentum_persistence / downside_momentum_clustering
        else:
            momentum_asymmetry_ratio = 0
            
        # Volatility-accelerated propagation
        if hist_2 is not None and hist_4 is not None and hist_9 is not None:
            momentum_acceleration = ((current['close'] - hist_2['close']) - (current['close'] - hist_4['close']) + 
                                   (current['close'] - hist_4['close']) - (current['close'] - hist_9['close']))
        else:
            momentum_acceleration = 0
            
        if hist_1 is not None and abs(current['open'] - hist_1['close']) > 0:
            volatility_efficiency = (current['high'] - current['low']) / abs(current['open'] - hist_1['close'])
        else:
            volatility_efficiency = 0
            
        acceleration_volatility_alignment = momentum_acceleration * volatility_efficiency
        
        # Microstructural Pressure Anchoring
        # Price discovery efficiency
        denominator_opening = max(current['high'] - current['low'], abs(current['open'] - hist_1['close'])) if hist_1 is not None else 1
        if denominator_opening > 0:
            opening_gap_efficiency = abs(current['close'] - current['open']) / denominator_opening
        else:
            opening_gap_efficiency = 0
            
        if current['high'] != current['low']:
            high_low_pressure_bias = (current['close'] - (current['high'] + current['low'])/2) / (current['high'] - current['low'])
        else:
            high_low_pressure_bias = 0
            
        if current['amount'] > 0:
            volume_weighted_momentum = (current['close'] - current['open']) * current['volume'] / current['amount']
        else:
            volume_weighted_momentum = 0
            
        # Order flow regime detection
        if current['high'] != current['low']:
            if current['close'] > current['open']:
                net_pressure_asymmetry = ((current['close'] - current['low']) / (current['high'] - current['low'])) * current['volume']
            elif current['close'] < current['open']:
                net_pressure_asymmetry = -((current['high'] - current['close']) / (current['high'] - current['low'])) * current['volume']
            else:
                net_pressure_asymmetry = 0
        else:
            net_pressure_asymmetry = 0
            
        # Volume intensity using only past data
        if len(volume_window) > 0:
            volume_intensity = current['volume'] / volume_window['volume'].mean()
        else:
            volume_intensity = 1
            
        # Microstructural momentum efficiency
        if current['volume'] > 0 and (current['high'] - current['low']) > 0:
            microstructural_momentum_efficiency = abs(current['close'] - current['open']) / np.sqrt(current['volume'] * (current['high'] - current['low']))
        else:
            microstructural_momentum_efficiency = 0
            
        # Cross-Regime Adaptive Framework
        # High volatility regime signals
        volatility_momentum_synthesis = momentum_asymmetry_ratio * acceleration_volatility_alignment
        pressure_efficiency = net_pressure_asymmetry * volume_intensity
        high_volatility_composite = volatility_momentum_synthesis * pressure_efficiency
        
        # Low volatility regime signals
        microstructural_alignment = opening_gap_efficiency * volume_weighted_momentum
        
        if current['high'] != current['low']:
            if current['close'] > current['open']:
                efficiency_spread = (current['close'] - current['open']) / (current['high'] - current['low'])
            elif current['close'] < current['open']:
                efficiency_spread = -abs(current['open'] - current['close']) / (current['high'] - current['low'])
            else:
                efficiency_spread = 0
        else:
            efficiency_spread = 0
            
        low_volatility_composite = microstructural_alignment * efficiency_spread
        
        # Transition regime signals
        # Breakout momentum using only past data
        high_window = df.iloc[max(0, i-20):i]['high']  # t-20 to t-1
        low_window = df.iloc[max(0, i-20):i]['low']    # t-20 to t-1
        
        if len(high_window) > 0 and len(low_window) > 0:
            max_high_prev = high_window.max()
            min_low_prev = low_window.min()
            breakout_up = max(current['close'] - max_high_prev, 0)
            breakout_down = max(min_low_prev - current['close'], 0)
            breakout_momentum = max(breakout_up, breakout_down) / (current['high'] - current['low']) if (current['high'] - current['low']) > 0 else 0
        else:
            breakout_momentum = 0
            
        acceleration_strength = (current['close'] - hist_4['close']) * momentum_acceleration if hist_4 is not None else 0
        transition_composite = breakout_momentum * acceleration_strength * microstructural_momentum_efficiency
        
        # Dynamic Signal Integration
        # Regime classification
        if len(vol_window) > 0:
            current_vol = (current['high'] - current['low']) / current['close']
            avg_vol = ((vol_window['high'] - vol_window['low']) / vol_window['close']).mean()
            regime_classification_volatility = current_vol / avg_vol if avg_vol > 0 else 1
        else:
            regime_classification_volatility = 1
            
        momentum_regime = np.sign(current['close'] - hist_4['close']) * momentum_asymmetry_ratio if hist_4 is not None else 0
        pressure_regime = net_pressure_asymmetry * volume_intensity
        
        # Composite Alpha Generation
        regime_adaptive_volatility_factor = high_volatility_composite * transition_composite * regime_classification_volatility
        microstructural_momentum_factor = low_volatility_composite * volume_weighted_momentum * high_low_pressure_bias
        
        # Final Adaptive Alpha
        final_alpha = regime_adaptive_volatility_factor * microstructural_momentum_factor * np.sign(momentum_acceleration) if momentum_acceleration != 0 else 0
        
        result.iloc[i] = final_alpha
        
    return result
