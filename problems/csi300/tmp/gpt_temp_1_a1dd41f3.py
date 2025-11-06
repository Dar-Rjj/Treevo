import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate Price-Volume Fractal Divergence Factor using multi-scale fractal analysis
    of price and volume patterns to identify regime shifts and divergence signals.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate price fractal dimensions
    def price_fractal_dimension(prices, window):
        """Calculate fractal dimension using Hurst exponent approximation"""
        if len(prices) < window:
            return np.nan
        
        # Calculate log returns
        returns = np.log(prices / prices.shift(1)).dropna()
        if len(returns) < window:
            return np.nan
            
        # Calculate rescaled range
        mean_return = returns.rolling(window=window).mean()
        deviations = returns - mean_return
        cumulative_deviations = deviations.rolling(window=window).sum()
        range_series = cumulative_deviations.rolling(window=window).max() - cumulative_deviations.rolling(window=window).min()
        std_series = returns.rolling(window=window).std()
        
        # Avoid division by zero
        valid_mask = (std_series > 0) & (range_series > 0)
        if not valid_mask.any():
            return np.nan
            
        rs_ratio = range_series / std_series
        rs_ratio = rs_ratio.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(rs_ratio) == 0:
            return np.nan
            
        # Hurst exponent approximation (fractal dimension = 2 - H)
        hurst = np.log(rs_ratio.mean()) / np.log(window)
        fractal_dim = 2 - hurst
        
        return fractal_dim
    
    # Calculate volume fractal dimensions
    def volume_fractal_dimension(volumes, window):
        """Calculate fractal dimension for volume series"""
        if len(volumes) < window:
            return np.nan
            
        # Normalize volume series
        volume_norm = volumes / volumes.rolling(window=window).mean()
        volume_norm = volume_norm.dropna()
        
        if len(volume_norm) < window:
            return np.nan
            
        # Calculate using similar approach as price fractal
        deviations = volume_norm - volume_norm.rolling(window=window).mean()
        cumulative_deviations = deviations.rolling(window=window).sum()
        range_series = cumulative_deviations.rolling(window=window).max() - cumulative_deviations.rolling(window=window).min()
        std_series = volume_norm.rolling(window=window).std()
        
        # Avoid division by zero
        valid_mask = (std_series > 0) & (range_series > 0)
        if not valid_mask.any():
            return np.nan
            
        rs_ratio = range_series / std_series
        rs_ratio = rs_ratio.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(rs_ratio) == 0:
            return np.nan
            
        hurst = np.log(rs_ratio.mean()) / np.log(window)
        fractal_dim = 2 - hurst
        
        return fractal_dim
    
    # Calculate multi-scale fractal dimensions for price
    price_fractal_3d = []
    price_fractal_8d = []
    
    for i in range(len(df)):
        if i >= 8:  # Need at least 8 days for calculation
            window_prices = df['close'].iloc[i-7:i+1]
            fractal_3d = price_fractal_dimension(window_prices, 3)
            fractal_8d = price_fractal_dimension(window_prices, 8)
            price_fractal_3d.append(fractal_3d)
            price_fractal_8d.append(fractal_8d)
        else:
            price_fractal_3d.append(np.nan)
            price_fractal_8d.append(np.nan)
    
    # Calculate multi-scale fractal dimensions for volume
    volume_fractal_3d = []
    volume_fractal_8d = []
    
    for i in range(len(df)):
        if i >= 8:  # Need at least 8 days for calculation
            window_volumes = df['volume'].iloc[i-7:i+1]
            fractal_3d = volume_fractal_dimension(window_volumes, 3)
            fractal_8d = volume_fractal_dimension(window_volumes, 8)
            volume_fractal_3d.append(fractal_3d)
            volume_fractal_8d.append(fractal_8d)
        else:
            volume_fractal_3d.append(np.nan)
            volume_fractal_8d.append(np.nan)
    
    # Calculate fractal ratios
    price_fractal_ratio = []
    volume_fractal_ratio = []
    
    for i in range(len(df)):
        if i >= 8:
            p_3d = price_fractal_3d[i]
            p_8d = price_fractal_8d[i]
            v_3d = volume_fractal_3d[i]
            v_8d = volume_fractal_8d[i]
            
            # Avoid division by zero and handle NaN values
            if pd.notna(p_3d) and pd.notna(p_8d) and p_8d != 0:
                price_ratio = p_3d / p_8d
            else:
                price_ratio = np.nan
                
            if pd.notna(v_3d) and pd.notna(v_8d) and v_8d != 0:
                volume_ratio = v_3d / v_8d
            else:
                volume_ratio = np.nan
                
            price_fractal_ratio.append(price_ratio)
            volume_fractal_ratio.append(volume_ratio)
        else:
            price_fractal_ratio.append(np.nan)
            volume_fractal_ratio.append(np.nan)
    
    # Calculate price-volume fractal divergence
    divergence_signals = []
    
    for i in range(len(df)):
        if i >= 8:
            price_ratio = price_fractal_ratio[i]
            volume_ratio = volume_fractal_ratio[i]
            
            if pd.notna(price_ratio) and pd.notna(volume_ratio):
                # Calculate divergence as the difference in fractal behavior
                divergence = price_ratio - volume_ratio
                
                # Calculate persistence (how long divergence has been maintained)
                if i >= 16:  # Need more data for persistence calculation
                    recent_divergences = []
                    for j in range(max(8, i-7), i+1):
                        if j < len(price_fractal_ratio) and j < len(volume_fractal_ratio):
                            p_r = price_fractal_ratio[j]
                            v_r = volume_fractal_ratio[j]
                            if pd.notna(p_r) and pd.notna(v_r):
                                recent_divergences.append(p_r - v_r)
                    
                    if len(recent_divergences) >= 5:
                        persistence = np.std(recent_divergences)  # Lower std = higher persistence
                        if persistence > 0:
                            # Weight by divergence magnitude and inverse persistence
                            weighted_signal = divergence * (1.0 / persistence)
                        else:
                            weighted_signal = divergence
                    else:
                        weighted_signal = divergence
                else:
                    weighted_signal = divergence
                    
                divergence_signals.append(weighted_signal)
            else:
                divergence_signals.append(np.nan)
        else:
            divergence_signals.append(np.nan)
    
    # Create final factor series
    for i in range(len(df)):
        if i >= 8 and i < len(divergence_signals):
            result.iloc[i] = divergence_signals[i]
        else:
            result.iloc[i] = np.nan
    
    return result
