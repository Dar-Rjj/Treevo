import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Nonlinear Micro-Fractal Alpha Framework
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(3, len(df)):
        current = df.iloc[i]
        prev1 = df.iloc[i-1] if i >= 1 else None
        prev2 = df.iloc[i-2] if i >= 2 else None
        prev3 = df.iloc[i-3] if i >= 3 else None
        prev7 = df.iloc[i-7] if i >= 7 else None
        
        # Avoid division by zero
        high_low_range = current['high'] - current['low']
        if high_low_range <= 0:
            continue
            
        # === Fractal Volatility Asymmetry ===
        if current['close'] > current['open']:
            up_fractal_vol = high_low_range * ((current['close'] - current['low']) / high_low_range) ** 2
            down_fractal_vol = 0
        elif current['close'] < current['open']:
            up_fractal_vol = 0
            down_fractal_vol = high_low_range * ((current['high'] - current['close']) / high_low_range) ** 2
        else:
            up_fractal_vol = 0
            down_fractal_vol = 0
            
        fractal_asymmetry_ratio = up_fractal_vol / (down_fractal_vol + 1e-8)
        
        # === Nonlinear Volume Dynamics ===
        if prev1 is not None:
            if current['volume'] > prev1['volume']:
                high_fractal_volume = current['volume'] * ((current['close'] - current['low']) ** 2) / high_low_range
                low_fractal_volume = 0
            elif current['volume'] < prev1['volume']:
                high_fractal_volume = 0
                low_fractal_volume = current['volume'] * ((current['high'] - current['close']) ** 2) / high_low_range
            else:
                high_fractal_volume = 0
                low_fractal_volume = 0
        else:
            high_fractal_volume = 0
            low_fractal_volume = 0
            
        volume_fractal_divergence = high_fractal_volume - low_fractal_volume
        
        # === Gap Fractal Momentum ===
        if prev1 is not None:
            if current['open'] > prev1['close']:
                up_gap_fractal = ((current['open'] - prev1['close']) ** 2) * (current['close'] - current['open']) / high_low_range
                down_gap_fractal = 0
            elif current['open'] < prev1['close']:
                up_gap_fractal = 0
                down_gap_fractal = ((prev1['close'] - current['open']) ** 2) * (current['open'] - current['close']) / high_low_range
            else:
                up_gap_fractal = 0
                down_gap_fractal = 0
        else:
            up_gap_fractal = 0
            down_gap_fractal = 0
            
        gap_fractal_asymmetry = up_gap_fractal - down_gap_fractal
        
        # === Intraday Fractal Pressure ===
        morning_fractal_pressure = ((current['high'] - current['open']) ** 2) * (current['close'] - current['low']) / high_low_range * current['volume']
        afternoon_fractal_pressure = ((current['close'] - current['low']) ** 2) * (current['high'] - current['close']) / high_low_range * current['volume']
        fractal_pressure_asymmetry = morning_fractal_pressure - afternoon_fractal_pressure
        
        # === Fractal Flow Dynamics ===
        if current['close'] > current['open']:
            buy_fractal_flow = current['amount'] * ((current['close'] - current['low']) ** 2) / high_low_range
            sell_fractal_flow = 0
        elif current['close'] < current['open']:
            buy_fractal_flow = 0
            sell_fractal_flow = current['amount'] * ((current['high'] - current['close']) ** 2) / high_low_range
        else:
            buy_fractal_flow = 0
            sell_fractal_flow = 0
            
        fractal_flow_asymmetry = buy_fractal_flow - sell_fractal_flow
        
        # === Multi-Fractal Asymmetry ===
        # Short-Fractal (3-day)
        if prev2 is not None:
            prev2_range = prev2['high'] - prev2['low']
            if prev2_range > 0:
                volatility_fractal = ((high_low_range ** 2) / prev2_range) * ((current['close'] - current['low']) - (current['high'] - current['close'])) / high_low_range
                volume_fractal = ((current['volume'] ** 2) / prev2['volume']) * ((current['close'] - current['low']) - (current['high'] - current['close'])) / high_low_range
            else:
                volatility_fractal = 0
                volume_fractal = 0
        else:
            volatility_fractal = 0
            volume_fractal = 0
            
        # Medium-Fractal (8-day)
        if prev7 is not None:
            price_range_fractal = ((current['high'] - prev7['close']) ** 2) / (prev7['close'] - current['low']) * ((current['close'] - current['low']) - (current['high'] - current['close'])) / high_low_range
            volume_range_fractal = ((current['volume'] ** 2) / prev7['volume']) * (current['close'] - current['open']) * ((current['close'] - current['low']) - (current['high'] - current['close'])) / high_low_range
        else:
            price_range_fractal = 0
            volume_range_fractal = 0
            
        cross_fractal_asymmetry = fractal_asymmetry_ratio * gap_fractal_asymmetry * fractal_pressure_asymmetry
        
        # === Regime-Fractal Adaptation ===
        # Momentum Fractal Regime
        if prev3 is not None and prev2 is not None and prev1 is not None:
            close_diff1 = current['close'] - prev1['close']
            close_diff2 = prev1['close'] - prev2['close']
            close_diff3 = prev2['close'] - prev3['close']
            
            if abs(close_diff2) > 1e-8 and abs(close_diff3) > 1e-8:
                fractal_acceleration = np.sign((close_diff1 ** 2) / close_diff2 - (close_diff2 ** 2) / close_diff3)
            else:
                fractal_acceleration = 0
                
            if prev2['volume'] > 0 and prev1['volume'] > 0:
                volume_fractal_regime = np.sign((current['volume'] ** 2) / prev1['volume'] - (prev1['volume'] ** 2) / prev2['volume'])
            else:
                volume_fractal_regime = 0
        else:
            fractal_acceleration = 0
            volume_fractal_regime = 0
            
        momentum_fractal_strength = fractal_acceleration * volume_fractal_regime
        
        # Volatility Fractal Regime
        if prev1 is not None:
            prev1_range = prev1['high'] - prev1['low']
            if prev1_range > 0 and prev1['volume'] > 0:
                range_fractal_expansion = ((high_low_range ** 2) / prev1_range) * (current['volume'] / prev1['volume'])
                asymmetry_fractal = (((current['high'] - current['close']) ** 2) - ((current['close'] - current['low']) ** 2)) / high_low_range * (high_low_range / prev1_range)
            else:
                range_fractal_expansion = 0
                asymmetry_fractal = 0
        else:
            range_fractal_expansion = 0
            asymmetry_fractal = 0
            
        # For volatility regime, we need previous day's range fractal expansion
        if i >= 4:
            prev_day = df.iloc[i-1]
            prev_prev1 = df.iloc[i-2]
            prev_prev1_range = prev_prev1['high'] - prev_prev1['low']
            if prev_prev1_range > 0 and prev_prev1['volume'] > 0:
                prev_range_fractal_expansion = ((prev_day['high'] - prev_day['low']) ** 2) / prev_prev1_range * (prev_day['volume'] / prev_prev1['volume'])
                volatility_fractal_regime = np.sign(range_fractal_expansion - prev_range_fractal_expansion)
            else:
                volatility_fractal_regime = 0
        else:
            volatility_fractal_regime = 0
            
        # Adaptive Fractal Weights
        momentum_fractal_weight = 1.6 if momentum_fractal_strength > 0 else 1.0
        volatility_fractal_weight = 1.4 if volatility_fractal_regime > 0 else 1.0
        fractal_regime_adaptation = momentum_fractal_weight * volatility_fractal_weight
        
        # === Composite Fractal Alpha ===
        core_fractal_asymmetry = fractal_asymmetry_ratio * fractal_flow_asymmetry * fractal_regime_adaptation
        momentum_fractal_asymmetry = gap_fractal_asymmetry * fractal_pressure_asymmetry * momentum_fractal_weight
        efficiency_fractal_asymmetry = volume_fractal_divergence * cross_fractal_asymmetry * volatility_fractal_weight
        
        final_alpha = core_fractal_asymmetry * momentum_fractal_asymmetry * efficiency_fractal_asymmetry
        
        result.iloc[i] = final_alpha
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
