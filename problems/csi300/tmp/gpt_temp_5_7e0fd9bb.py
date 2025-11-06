import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(19, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Multi-Scale Fractal Analysis
        # Short-term fractal dimension (5-day window)
        if i >= 4:
            short_high_low = current_data['high'].iloc[i-4:i+1].max() - current_data['low'].iloc[i-4:i+1].min()
            short_std = current_data['close'].iloc[i-4:i+1].std()
            short_fractal = (np.log(max(short_high_low, 1e-6)) / np.log(5)) - (np.log(max(short_std, 1e-6)) / np.log(5))
        else:
            short_fractal = np.nan
            
        # Medium-term fractal dimension (15-day window)
        if i >= 14:
            medium_high_low = current_data['high'].iloc[i-14:i+1].max() - current_data['low'].iloc[i-14:i+1].min()
            medium_std = current_data['close'].iloc[i-14:i+1].std()
            medium_fractal = (np.log(max(medium_high_low, 1e-6)) / np.log(15)) - (np.log(max(medium_std, 1e-6)) / np.log(15))
        else:
            medium_fractal = np.nan
            
        # Fractal Dimension Ratio
        if not np.isnan(short_fractal) and not np.isnan(medium_fractal) and medium_fractal != 0:
            fractal_ratio = short_fractal / medium_fractal
        else:
            fractal_ratio = np.nan
            
        # Price-Volume Fractal Divergence (simplified)
        if i >= 4:
            price_range = current_data['high'].iloc[i-4:i+1].max() - current_data['low'].iloc[i-4:i+1].min()
            volume_range = current_data['volume'].iloc[i-4:i+1].max() - current_data['volume'].iloc[i-4:i+1].min()
            if price_range > 0 and volume_range > 0:
                price_vol_divergence = (np.log(price_range) - np.log(volume_range)) / 5
            else:
                price_vol_divergence = 0
        else:
            price_vol_divergence = 0
            
        # Market State Classification
        market_state = "normal"
        if not np.isnan(fractal_ratio):
            if fractal_ratio < 0.9:
                market_state = "trending"
            elif fractal_ratio > 1.1:
                market_state = "mean_reverting"
            if abs(price_vol_divergence) > 0.2:
                market_state = "chaotic"
        
        # Turnover-Momentum Divergence Analysis
        # Momentum Components
        if i >= 5:
            short_momentum = (current_data['close'].iloc[i] - current_data['close'].iloc[i-5]) / max(current_data['high'].iloc[i-4:i+1].max() - current_data['low'].iloc[i-4:i+1].min(), 1e-6)
        else:
            short_momentum = 0
            
        if i >= 15:
            medium_momentum = (current_data['close'].iloc[i] - current_data['close'].iloc[i-15]) / max(current_data['high'].iloc[i-14:i+1].max() - current_data['low'].iloc[i-14:i+1].min(), 1e-6)
        else:
            medium_momentum = 0
            
        momentum_divergence = short_momentum - medium_momentum
        
        # Turnover Components
        if i >= 4:
            turnover_5d = np.mean([current_data['volume'].iloc[j] * current_data['close'].iloc[j] for j in range(i-4, i+1)])
        else:
            turnover_5d = current_data['volume'].iloc[i] * current_data['close'].iloc[i]
            
        if i >= 14:
            turnover_15d = np.mean([current_data['volume'].iloc[j] * current_data['close'].iloc[j] for j in range(i-14, i+1)])
        else:
            turnover_15d = turnover_5d
            
        turnover_ratio = (turnover_5d / max(turnover_15d, 1e-6)) - 1
        
        divergence_signal = momentum_divergence * turnover_ratio
        
        # Volume-Price Asymmetry Analysis
        # Upside Volume Confirmation (10-day window)
        if i >= 9:
            up_days_volume = []
            total_volume = []
            for j in range(i-9, i+1):
                if j > 0 and current_data['close'].iloc[j] > current_data['close'].iloc[j-1]:
                    up_days_volume.append(current_data['volume'].iloc[j])
                total_volume.append(current_data['volume'].iloc[j])
            
            up_volume_avg = np.mean(up_days_volume) if up_days_volume else 0
            total_volume_avg = np.mean(total_volume)
            up_volume_ratio = up_volume_avg / max(total_volume_avg, 1e-6)
        else:
            up_volume_ratio = 0.5
            
        # Price Movement Asymmetry (10-day window)
        if i >= 9:
            positive_returns = 0
            negative_returns = 0
            for j in range(i-9, i+1):
                if j > 0:
                    ret = current_data['close'].iloc[j] / current_data['close'].iloc[j-1] - 1
                    if ret > 0:
                        positive_returns += ret
                    else:
                        negative_returns += abs(ret)
            
            price_asymmetry = np.log(1 + positive_returns) - np.log(1 + negative_returns)
        else:
            price_asymmetry = 0
            
        asymmetry_signal = up_volume_ratio * price_asymmetry
        
        # Base Component
        base_component = divergence_signal * asymmetry_signal
        
        # Volatility-Adaptive Synthesis
        # True Range Volatility (20-day window)
        if i >= 19:
            true_ranges = []
            for j in range(i-19, i+1):
                if j > 0:
                    tr1 = current_data['high'].iloc[j] - current_data['low'].iloc[j]
                    tr2 = abs(current_data['high'].iloc[j] - current_data['close'].iloc[j-1])
                    tr3 = abs(current_data['low'].iloc[j] - current_data['close'].iloc[j-1])
                    true_ranges.append(max(tr1, tr2, tr3))
            true_range_vol = np.mean(true_ranges) if true_ranges else 1
        else:
            true_range_vol = 1
            
        # High-Low Range Component (20-day window)
        if i >= 19:
            high_max = current_data['high'].iloc[i-19:i+1].max()
            low_min = current_data['low'].iloc[i-19:i+1].min()
            high_low_component = (high_max / max(low_min, 1e-6)) - 1
        else:
            high_low_component = 1
            
        # Market-State Adaptive Combination
        if market_state == "trending" and not np.isnan(fractal_ratio):
            factor_value = base_component * fractal_ratio / max(true_range_vol, 1e-6)
        elif market_state == "mean_reverting" and not np.isnan(fractal_ratio):
            factor_value = base_component / (max(fractal_ratio, 1e-6) * max(true_range_vol, 1e-6))
        elif market_state == "chaotic":
            factor_value = base_component / (max(abs(price_vol_divergence), 1e-6) * max(high_low_component, 1e-6))
        else:
            factor_value = base_component / max(true_range_vol, 1e-6)
        
        result.iloc[i] = factor_value
    
    return result.fillna(0)
