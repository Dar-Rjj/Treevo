import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Cross-Scale Entropy & Fractal Market Dynamics factor
    Combines multi-fractal price structure, volume-entropy co-evolution,
    amount-flow characteristics, cross-scale information asymmetry, and
    market microstructure fractals.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need sufficient data for calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # Multi-Fractal Price Structure Analysis
        hurst_factors = []
        fractal_dims = []
        
        # Hurst exponent estimation for different windows
        for window in [5, 10, 20]:
            if i >= window:
                window_data = current_data.iloc[-window:]
                returns = np.log(window_data['close'] / window_data['close'].shift(1)).dropna()
                
                if len(returns) > 1:
                    # Rescaled range analysis for Hurst estimation
                    mean_return = returns.mean()
                    deviations = returns - mean_return
                    cumulative_deviations = deviations.cumsum()
                    r_range = cumulative_deviations.max() - cumulative_deviations.min()
                    std_dev = returns.std()
                    
                    if std_dev > 0:
                        hurst = np.log(r_range / std_dev) / np.log(len(returns))
                        hurst_factors.append(hurst)
        
        # Fractal dimension using high-low range
        if i >= 10:
            recent_data = current_data.iloc[-10:]
            ranges = (recent_data['high'] - recent_data['low']) / recent_data['close']
            if ranges.std() > 0:
                # Box-counting inspired fractal dimension estimation
                fractal_dim = 2 - (np.log(ranges.var()) / np.log(10))
                fractal_dims.append(fractal_dim)
        
        # Volume-Entropy Co-evolution
        volume_entropies = []
        for window in [3, 5, 10]:
            if i >= window:
                volume_data = current_data['volume'].iloc[-window:]
                # Shannon entropy of volume distribution
                volume_norm = volume_data / volume_data.sum()
                entropy = -np.sum(volume_norm * np.log(volume_norm + 1e-10))
                volume_entropies.append(entropy)
        
        # Amount-Flow Fractal Characteristics
        amount_factors = []
        if i >= 15:
            amount_data = current_data['amount'].iloc[-15:]
            # Heavy-tailedness measurement using log-log relationship
            sorted_amount = np.sort(amount_data)[::-1]
            ranks = np.arange(1, len(sorted_amount) + 1)
            
            if len(sorted_amount) > 5:
                # Power law exponent estimation
                log_ranks = np.log(ranks)
                log_amounts = np.log(sorted_amount + 1e-10)
                slope, _, _, _, _ = linregress(log_ranks, log_amounts)
                amount_factors.append(-slope)  # Negative for heavy-tailedness
        
        # Cross-Scale Information Asymmetry
        cross_scale_factors = []
        
        # Intraday vs multi-day volatility ratio
        if i >= 10:
            recent_data = current_data.iloc[-10:]
            intraday_vol = (recent_data['high'] - recent_data['low']) / recent_data['close']
            multiday_vol = np.abs(np.log(recent_data['close'] / recent_data['close'].shift(1))).rolling(3).mean()
            
            if multiday_vol.iloc[-1] > 0:
                vol_ratio = intraday_vol.iloc[-1] / multiday_vol.iloc[-1]
                cross_scale_factors.append(vol_ratio)
        
        # Overnight gap efficiency
        if i >= 2:
            prev_close = current_data['close'].iloc[-2]
            curr_open = current_data['open'].iloc[-1]
            prev_range = current_data['high'].iloc[-2] - current_data['low'].iloc[-2]
            
            if prev_range > 0:
                gap_efficiency = abs(curr_open - prev_close) / prev_range
                cross_scale_factors.append(gap_efficiency)
        
        # Market Microstructure Fractals
        micro_factors = []
        
        # Bid-ask spread implied volatility (using high-low range)
        if i >= 5:
            recent_data = current_data.iloc[-5:]
            spread_vol = (recent_data['high'] - recent_data['low']) / recent_data['close']
            micro_factors.append(spread_vol.mean())
        
        # Price impact curvature estimation
        if i >= 8:
            recent_data = current_data.iloc[-8:]
            volume_changes = np.diff(recent_data['volume'])
            price_changes = np.diff(np.log(recent_data['close']))
            
            if len(volume_changes) > 2 and len(price_changes) > 2:
                # Simple curvature estimation using second differences
                if np.std(volume_changes) > 0:
                    impact_curvature = np.std(np.diff(price_changes / (volume_changes + 1e-10)))
                    micro_factors.append(impact_curvature)
        
        # Combine all sub-factors with appropriate weights
        combined_factor = 0
        weight_sum = 0
        
        # Hurst factors (trend persistence)
        if hurst_factors:
            hurst_avg = np.mean(hurst_factors)
            combined_factor += (hurst_avg - 0.5) * 2.0  # Center around 0.5
            weight_sum += 2.0
        
        # Fractal dimensions (market complexity)
        if fractal_dims:
            fractal_avg = np.mean(fractal_dims)
            combined_factor += (fractal_avg - 1.5) * 1.5  # Center around 1.5
            weight_sum += 1.5
        
        # Volume entropy (information content)
        if volume_entropies:
            entropy_avg = np.mean(volume_entropies)
            combined_factor += entropy_avg * 1.2
            weight_sum += 1.2
        
        # Amount flow characteristics (large transaction patterns)
        if amount_factors:
            amount_avg = np.mean(amount_factors)
            combined_factor += amount_avg * 1.0
            weight_sum += 1.0
        
        # Cross-scale factors (information asymmetry)
        if cross_scale_factors:
            cross_avg = np.mean(cross_scale_factors)
            combined_factor += cross_avg * 1.3
            weight_sum += 1.3
        
        # Microstructure factors (trading dynamics)
        if micro_factors:
            micro_avg = np.mean(micro_factors)
            combined_factor += micro_avg * 0.8
            weight_sum += 0.8
        
        if weight_sum > 0:
            result.iloc[i] = combined_factor / weight_sum
        else:
            result.iloc[i] = 0
    
    return result
