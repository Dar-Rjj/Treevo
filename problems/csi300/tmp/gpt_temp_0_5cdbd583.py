import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Efficiency with Regime-Sensitive Mean Reversion
    """
    data = df.copy()
    
    # Price Path Efficiency Ratio
    def price_path_efficiency(window):
        if len(window) < 2:
            return np.nan
        actual_distance = np.sum(np.abs(np.diff(window)))
        straight_distance = np.abs(window[-1] - window[0])
        return straight_distance / actual_distance if actual_distance > 0 else 0
    
    # Volume-weighted price oscillations
    def volume_weighted_oscillation(window_prices, window_volumes):
        if len(window_prices) < 2:
            return np.nan
        price_changes = np.abs(np.diff(window_prices))
        weighted_volumes = window_volumes[:-1] + window_volumes[1:]
        return np.sum(price_changes * weighted_volumes) / np.sum(weighted_volumes) if np.sum(weighted_volumes) > 0 else 0
    
    # Multi-scale efficiency calculation
    short_window = 5
    long_window = 20
    
    # Short-term efficiency (5 days)
    short_efficiency = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= short_window - 1:
            window_prices = data['close'].iloc[i-short_window+1:i+1].values
            short_efficiency.iloc[i] = price_path_efficiency(window_prices)
    
    # Long-term efficiency (20 days)
    long_efficiency = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= long_window - 1:
            window_prices = data['close'].iloc[i-long_window+1:i+1].values
            long_efficiency.iloc[i] = price_path_efficiency(window_prices)
    
    # Volume fractal dimension approximation
    volume_fractal = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 9:  # 10-day window for volume fractal
            window_volumes = data['volume'].iloc[i-9:i+1].values
            if np.sum(window_volumes) > 0:
                # Simple fractal dimension approximation using volume variance
                volume_range = np.max(window_volumes) - np.min(window_volumes)
                volume_mean = np.mean(window_volumes)
                volume_fractal.iloc[i] = 1 + (np.log(volume_range) / np.log(volume_mean)) if volume_mean > 0 else 1.0
    
    # Efficiency-based regime classification
    regime = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if pd.notna(short_efficiency.iloc[i]) and pd.notna(long_efficiency.iloc[i]):
            # High efficiency = trending, low efficiency = mean-reverting
            eff_ratio = short_efficiency.iloc[i] / long_efficiency.iloc[i] if long_efficiency.iloc[i] > 0 else 1.0
            regime.iloc[i] = 1.0 if eff_ratio > 1.1 else -1.0 if eff_ratio < 0.9 else 0.0
    
    # Price deviation from efficiency trend
    price_deviation = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:  # 5-day window
            recent_prices = data['close'].iloc[i-4:i+1].values
            if len(recent_prices) >= 2:
                price_trend = recent_prices[-1] - recent_prices[0]
                price_volatility = np.std(recent_prices)
                price_deviation.iloc[i] = price_trend / price_volatility if price_volatility > 0 else 0
    
    # Volume-confirmed reversal patterns
    volume_confirmation = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 2:  # 3-day window
            current_volume = data['volume'].iloc[i]
            prev_volume = data['volume'].iloc[i-1]
            volume_ratio = current_volume / prev_volume if prev_volume > 0 else 1.0
            price_change = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1] if data['close'].iloc[i-1] > 0 else 0
            
            # Volume confirmation: high volume with price reversal
            if abs(price_change) > 0.02:  # 2% price move
                volume_confirmation.iloc[i] = volume_ratio * (-1 if price_change > 0 else 1)
            else:
                volume_confirmation.iloc[i] = 0
    
    # Adaptive mean reversion strength
    reversion_strength = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if pd.notna(price_deviation.iloc[i]) and pd.notna(volume_confirmation.iloc[i]):
            # Stronger reversion in low-efficiency regimes
            regime_multiplier = 2.0 if regime.iloc[i] < 0 else 0.5 if regime.iloc[i] > 0 else 1.0
            reversion_strength.iloc[i] = price_deviation.iloc[i] * volume_confirmation.iloc[i] * regime_multiplier
    
    # Fractal consistency scoring
    fractal_consistency = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if pd.notna(short_efficiency.iloc[i]) and pd.notna(long_efficiency.iloc[i]) and pd.notna(volume_fractal.iloc[i]):
            # Multi-timeframe efficiency alignment
            eff_alignment = 1.0 - abs(short_efficiency.iloc[i] - long_efficiency.iloc[i])
            
            # Volume-price fractal coherence (simplified)
            volume_coherence = 1.0 / (1.0 + abs(volume_fractal.iloc[i] - 1.5))  # Target fractal dimension ~1.5
            
            fractal_consistency.iloc[i] = eff_alignment * volume_coherence
    
    # Final factor integration
    factor = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if (pd.notna(reversion_strength.iloc[i]) and 
            pd.notna(fractal_consistency.iloc[i]) and 
            pd.notna(regime.iloc[i])):
            
            # Regime-specific weighting
            if regime.iloc[i] < 0:  # Mean-reverting regime
                factor_weight = 0.7
            elif regime.iloc[i] > 0:  # Trending regime  
                factor_weight = 0.3
            else:  # Neutral regime
                factor_weight = 0.5
            
            # Dynamic position sizing based on fractal consistency
            position_scale = fractal_consistency.iloc[i]
            
            factor.iloc[i] = reversion_strength.iloc[i] * factor_weight * position_scale
    
    return factor.fillna(0)
