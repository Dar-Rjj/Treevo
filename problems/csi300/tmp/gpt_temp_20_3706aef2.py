import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Pressure-Efficiency Momentum with Volume-Price Coherence alpha factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Helper function for local extrema detection (5-point fractal pattern)
    def detect_fractals(high_series, low_series):
        highs, lows = [], []
        for i in range(2, len(high_series)-2):
            # Bullish fractal (low point)
            if (low_series.iloc[i] < low_series.iloc[i-2] and 
                low_series.iloc[i] < low_series.iloc[i-1] and 
                low_series.iloc[i] < low_series.iloc[i+1] and 
                low_series.iloc[i] < low_series.iloc[i+2]):
                lows.append((high_series.index[i], 'low', low_series.iloc[i]))
            # Bearish fractal (high point)
            if (high_series.iloc[i] > high_series.iloc[i-2] and 
                high_series.iloc[i] > high_series.iloc[i-1] and 
                high_series.iloc[i] > high_series.iloc[i+1] and 
                high_series.iloc[i] > high_series.iloc[i+2]):
                highs.append((high_series.index[i], 'high', high_series.iloc[i]))
        return highs + lows
    
    # Calculate intraday pressure (morning vs afternoon strength)
    def intraday_pressure(row):
        morning_range = (row['high'] - row['open']) / (row['high'] - row['low'] + 1e-8)
        afternoon_range = (row['close'] - row['low']) / (row['high'] - row['low'] + 1e-8)
        return morning_range - afternoon_range
    
    # Calculate daily efficiency (price path efficiency)
    def daily_efficiency(row):
        price_range = row['high'] - row['low']
        if price_range == 0:
            return 0
        actual_path = abs(row['close'] - row['open'])
        return actual_path / price_range
    
    # Calculate trade size efficiency
    def trade_size_efficiency(row):
        if row['volume'] == 0:
            return 0
        return row['amount'] / row['volume'] / row['close']
    
    # Initialize rolling calculations
    window_10 = 10
    window_3 = 3
    
    for i in range(window_10, len(df)):
        current_data = df.iloc[:i+1]
        
        # 1. Fractal Pattern Analysis
        fractals = detect_fractals(current_data['high'], current_data['low'])
        
        # Intraday pressure at fractal points
        fractal_pressures = []
        for date, f_type, price in fractals:
            if date in current_data.index:
                pressure = intraday_pressure(current_data.loc[date])
                fractal_pressures.append((date, f_type, price, pressure))
        
        # Multi-timeframe fractal density
        recent_fractals = [f for f in fractals if f[0] >= current_data.index[i-9]]
        fractal_density = len(recent_fractals) / window_10
        
        # Pressure-weighted fractal density
        recent_pressures = [p[3] for p in fractal_pressures if p[0] >= current_data.index[i-9]]
        pressure_weighted_density = fractal_density * (np.mean(recent_pressures) if recent_pressures else 0)
        
        # Efficiency-adjusted fractal strength
        current_eff = daily_efficiency(current_data.iloc[i])
        efficiency_adjusted_strength = pressure_weighted_density * current_eff
        
        # Fractal boundary momentum
        if len(fractals) >= 2:
            recent_fractals_sorted = sorted([f for f in fractals if f[0] <= current_data.index[i]], 
                                          key=lambda x: x[0])
            upper_fractals = [f for f in recent_fractals_sorted if f[1] == 'high']
            lower_fractals = [f for f in recent_fractals_sorted if f[1] == 'low']
            
            if upper_fractals and lower_fractals:
                nearest_upper = max(upper_fractals, key=lambda x: x[0])
                nearest_lower = max(lower_fractals, key=lambda x: x[0])
                current_price = current_data.iloc[i]['close']
                
                if nearest_upper[2] != nearest_lower[2]:
                    boundary_momentum = (current_price - nearest_lower[2]) / (nearest_upper[2] - nearest_lower[2])
                else:
                    boundary_momentum = 0.5
            else:
                boundary_momentum = 0.5
        else:
            boundary_momentum = 0.5
        
        # 2. Volume-Price Synchronization
        # Volume concentration at fractal points
        fractal_volumes = []
        for date, f_type, price in fractals:
            if date in current_data.index:
                vol = current_data.loc[date, 'volume']
                fractal_volumes.append(vol)
        
        current_volume = current_data.iloc[i]['volume']
        if fractal_volumes:
            volume_concentration = current_volume / np.mean(fractal_volumes[-5:]) if len(fractal_volumes) >= 5 else 1
        else:
            volume_concentration = 1
        
        # Volume-pressure coherence
        current_pressure = intraday_pressure(current_data.iloc[i])
        volume_pressure_coherence = 1 if np.sign(current_volume - current_data.iloc[i-1]['volume']) == np.sign(current_pressure) else -1
        
        # Trade size efficiency alignment
        current_trade_eff = trade_size_efficiency(current_data.iloc[i])
        avg_trade_eff = np.mean([trade_size_efficiency(current_data.iloc[j]) for j in range(max(0, i-4), i+1)])
        trade_size_pattern = current_trade_eff / (avg_trade_eff + 1e-8)
        
        # Volume-fractal momentum
        volume_fractal_momentum = volume_pressure_coherence * boundary_momentum
        
        # Multi-timeframe volume-fractal consistency
        short_term_vol = np.mean([current_data.iloc[j]['volume'] for j in range(max(0, i-2), i+1)])
        long_term_vol = np.mean([current_data.iloc[j]['volume'] for j in range(max(0, i-9), i+1)])
        
        if long_term_vol > 0:
            multi_timeframe_consistency = short_term_vol / long_term_vol
        else:
            multi_timeframe_consistency = 1
        
        # Fractal volume efficiency
        path_efficiency_ratio = current_eff / (np.mean([daily_efficiency(current_data.iloc[j]) for j in range(max(0, i-4), i+1)]) + 1e-8)
        fractal_volume_efficiency = volume_fractal_momentum * path_efficiency_ratio
        
        # 3. Pressure-Efficiency Divergence
        # Fractal divergence strength
        pressure_divergence = current_pressure - np.mean([intraday_pressure(current_data.iloc[j]) for j in range(max(0, i-4), i)])
        fractal_divergence_strength = abs(pressure_divergence) * fractal_density
        
        # Fractal regime score
        fractal_regime_score = current_eff * pressure_weighted_density
        
        # Volume-supported fractal breaks
        volume_supported_breaks = boundary_momentum * volume_concentration
        
        # Trade size confirmation
        trade_size_confirmation = trade_size_pattern * volume_fractal_momentum
        
        # Fractal pressure amplification
        fractal_pressure_amplification = 1 + volume_supported_breaks * trade_size_confirmation
        
        # 4. Adaptive Composite Factor Generation
        # Core fractal-pressure momentum
        base_fractal_component = efficiency_adjusted_strength * boundary_momentum
        pressure_divergence_enhanced = base_fractal_component * fractal_divergence_strength
        volume_coherence_multiplier = 1 + volume_pressure_coherence * fractal_volume_efficiency
        
        # Regime-based fractal filtering
        efficiency_volatility_filter = pressure_divergence_enhanced * fractal_regime_score
        volume_concentrated_boost = efficiency_volatility_filter * fractal_pressure_amplification
        multi_timeframe_boost = volume_concentrated_boost * multi_timeframe_consistency
        
        # Final alpha factor
        raw_fractal_composite = multi_timeframe_boost * volume_coherence_multiplier
        fractal_direction_alignment = raw_fractal_composite * np.sign(pressure_divergence)
        final_factor = fractal_direction_alignment * trade_size_pattern
        
        result.iloc[i] = final_factor
    
    # Fill initial values with 0
    result = result.fillna(0)
    
    return result
