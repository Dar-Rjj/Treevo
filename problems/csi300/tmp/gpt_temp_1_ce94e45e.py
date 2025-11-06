import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Asymmetry with Fractal Efficiency alpha factor
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Fractal Market Efficiency Components
    # Short-Term Fractal Dimension (5-day)
    def short_term_fractal_ratio(close, volume, window=5):
        price_efficiency = pd.Series(index=close.index, dtype=float)
        volume_efficiency = pd.Series(index=volume.index, dtype=float)
        
        for i in range(window-1, len(close)):
            if i >= window-1:
                # Price path efficiency
                net_move = close.iloc[i] - close.iloc[i-(window-1)]
                total_path = sum(abs(close.iloc[j] - close.iloc[j-1]) for j in range(i-(window-2), i+1))
                price_efficiency.iloc[i] = net_move / total_path if total_path != 0 else 0
                
                # Volume path efficiency
                vol_net_move = volume.iloc[i] - volume.iloc[i-(window-1)]
                vol_total_path = sum(abs(volume.iloc[j] - volume.iloc[j-1]) for j in range(i-(window-2), i+1))
                volume_efficiency.iloc[i] = vol_net_move / vol_total_path if vol_total_path != 0 else 0
        
        fractal_ratio = price_efficiency / volume_efficiency.replace(0, np.nan)
        return fractal_ratio.fillna(0)
    
    # Medium-Term Fractal Dimension (20-day)
    def medium_term_fractal_ratio(close, volume, window=20):
        price_efficiency = pd.Series(index=close.index, dtype=float)
        volume_efficiency = pd.Series(index=volume.index, dtype=float)
        
        for i in range(window-1, len(close)):
            if i >= window-1:
                # Price path efficiency
                net_move = close.iloc[i] - close.iloc[i-(window-1)]
                total_path = sum(abs(close.iloc[j] - close.iloc[j-1]) for j in range(i-(window-2), i+1))
                price_efficiency.iloc[i] = net_move / total_path if total_path != 0 else 0
                
                # Volume path efficiency
                vol_net_move = volume.iloc[i] - volume.iloc[i-(window-1)]
                vol_total_path = sum(abs(volume.iloc[j] - volume.iloc[j-1]) for j in range(i-(window-2), i+1))
                volume_efficiency.iloc[i] = vol_net_move / vol_total_path if vol_total_path != 0 else 0
        
        fractal_ratio = price_efficiency / volume_efficiency.replace(0, np.nan)
        return fractal_ratio.fillna(0)
    
    # Calculate fractal ratios
    short_fractal = short_term_fractal_ratio(data['close'], data['volume'], 5)
    medium_fractal = medium_term_fractal_ratio(data['close'], data['volume'], 20)
    
    # Efficiency Gap Analysis
    efficiency_gap = short_fractal - medium_fractal
    efficiency_acceleration = efficiency_gap / medium_fractal.replace(0, np.nan)
    efficiency_acceleration = efficiency_acceleration.fillna(0)
    
    # Asymmetric Volume-Price Response
    # Up-Day vs Down-Day Analysis
    def volume_asymmetry(close, volume, window=10):
        asymmetry = pd.Series(index=close.index, dtype=float)
        
        for i in range(window-1, len(close)):
            window_data = close.iloc[i-(window-1):i+1]
            window_volume = volume.iloc[i-(window-1):i+1]
            
            up_volume = 0
            down_volume = 0
            total_volume = window_volume.sum()
            
            for j in range(1, len(window_data)):
                if window_data.iloc[j] > window_data.iloc[j-1]:
                    up_volume += window_volume.iloc[j]
                elif window_data.iloc[j] < window_data.iloc[j-1]:
                    down_volume += window_volume.iloc[j]
            
            if total_volume > 0:
                up_intensity = up_volume / total_volume
                down_intensity = down_volume / total_volume
                asymmetry.iloc[i] = (up_intensity - down_intensity) / (up_intensity + down_intensity + 1e-8)
            else:
                asymmetry.iloc[i] = 0
        
        return asymmetry
    
    vol_asymmetry = volume_asymmetry(data['close'], data['volume'])
    
    # Price Response to Volume Extremes
    def volume_sensitivity_gap(close, volume, window=20):
        sensitivity_gap = pd.Series(index=close.index, dtype=float)
        
        for i in range(window, len(close)):
            vol_window = volume.iloc[i-window:i]
            high_vol_threshold = vol_window.quantile(0.9)
            low_vol_threshold = vol_window.quantile(0.1)
            
            current_vol = volume.iloc[i]
            current_ret = close.iloc[i] / close.iloc[i-1] - 1
            
            if current_vol > high_vol_threshold:
                high_vol_ret = current_ret
            else:
                high_vol_ret = 0
                
            if current_vol < low_vol_threshold:
                low_vol_ret = current_ret
            else:
                low_vol_ret = 0
            
            sensitivity_gap.iloc[i] = high_vol_ret - low_vol_ret
        
        return sensitivity_gap
    
    vol_sensitivity_gap = volume_sensitivity_gap(data['close'], data['volume'])
    
    # Market Microstructure Patterns
    def microstructure_pressure(open_p, high, low, close):
        pressure_ratio = pd.Series(index=open_p.index, dtype=float)
        
        for i in range(1, len(open_p)):
            opening_gap = open_p.iloc[i] - close.iloc[i-1]
            daily_range = high.iloc[i] - low.iloc[i]
            
            if daily_range != 0:
                opening_efficiency = opening_gap / daily_range
            else:
                opening_efficiency = 0
            
            closing_momentum = close.iloc[i] - open_p.iloc[i]
            if daily_range != 0:
                closing_efficiency = closing_momentum / daily_range
            else:
                closing_efficiency = 0
            
            if opening_efficiency != 0:
                pressure_ratio.iloc[i] = closing_efficiency / opening_efficiency
            else:
                pressure_ratio.iloc[i] = 0
        
        return pressure_ratio
    
    microstructure_pressure_ratio = microstructure_pressure(data['open'], data['high'], data['low'], data['close'])
    
    # High-Low Range Utilization
    def range_efficiency(open_p, high, low, close, window=5):
        range_eff = pd.Series(index=open_p.index, dtype=float)
        range_consistency = pd.Series(index=open_p.index, dtype=float)
        
        for i in range(window-1, len(open_p)):
            # Daily range efficiency
            daily_range = high.iloc[i] - low.iloc[i]
            if daily_range != 0:
                range_eff.iloc[i] = (close.iloc[i] - open_p.iloc[i]) / daily_range
            else:
                range_eff.iloc[i] = 0
            
            # Range consistency
            range_window = [high.iloc[j] - low.iloc[j] for j in range(i-(window-1), i+1)]
            if np.mean(range_window) != 0:
                range_consistency.iloc[i] = np.std(range_window) / np.mean(range_window)
            else:
                range_consistency.iloc[i] = 0
        
        return range_eff, range_consistency
    
    range_efficiency_ratio, range_consistency = range_efficiency(data['open'], data['high'], data['low'], data['close'])
    
    # Multi-Scale Convergence Detection
    def multi_scale_convergence(short_fractal, medium_fractal, window=10):
        convergence = pd.Series(index=short_fractal.index, dtype=float)
        
        for i in range(window-1, len(short_fractal)):
            short_window = short_fractal.iloc[i-(window-1):i+1]
            medium_window = medium_fractal.iloc[i-(window-1):i+1]
            
            # Correlation between short and medium term
            if len(short_window) > 1 and len(medium_window) > 1:
                corr = np.corrcoef(short_window, medium_window)[0,1]
                if np.isnan(corr):
                    corr = 0
            else:
                corr = 0
            
            # Direction alignment
            short_dir = 1 if short_fractal.iloc[i] > short_fractal.iloc[i-1] else -1
            medium_dir = 1 if medium_fractal.iloc[i] > medium_fractal.iloc[i-1] else -1
            
            direction_score = 1 if short_dir == medium_dir else -1
            
            convergence.iloc[i] = corr * direction_score
        
        return convergence
    
    convergence_score = multi_scale_convergence(short_fractal, medium_fractal)
    
    # Recent price volatility for scaling
    def recent_volatility(close, window=10):
        returns = close.pct_change().fillna(0)
        volatility = returns.rolling(window=window, min_periods=1).std().fillna(0)
        return volatility
    
    price_volatility = recent_volatility(data['close'])
    
    # Adaptive Signal Generation
    for i in range(len(data)):
        if i >= 20:  # Ensure sufficient data for calculations
            # Fractal Efficiency Scoring
            fractal_score = (short_fractal.iloc[i] + medium_fractal.iloc[i]) / 2
            fractal_score *= (1 + efficiency_acceleration.iloc[i])
            
            # Asymmetry Integration
            asymmetry_component = vol_asymmetry.iloc[i] * (1 + vol_sensitivity_gap.iloc[i])
            asymmetry_component *= (1 + microstructure_pressure_ratio.iloc[i])
            asymmetry_component *= (1 + range_efficiency_ratio.iloc[i])
            
            # Final Alpha Construction
            base_signal = fractal_score * asymmetry_component
            scaled_signal = base_signal * (1 + convergence_score.iloc[i])
            
            # Apply volatility scaling (inverse relationship)
            if price_volatility.iloc[i] > 0:
                volatility_scaling = 1 / (1 + price_volatility.iloc[i])
            else:
                volatility_scaling = 1
                
            alpha.iloc[i] = scaled_signal * volatility_scaling
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
