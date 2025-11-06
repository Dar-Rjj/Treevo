import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Price-Volume Fractal Dynamics factor
    Combines fractal momentum efficiency, volume fractal dimension, 
    and price-volume fractal correlation across multiple time scales
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Define multiple time scales for fractal analysis
    scales = [5, 10, 20, 40]
    
    for i in range(max(scales), len(df)):
        current_data = df.iloc[:i+1]
        
        # Fractal Momentum Efficiency Component
        momentum_efficiency = 0
        for scale in scales:
            if i >= scale:
                # Calculate price path efficiency (Hurst-like exponent)
                price_range = current_data['high'].iloc[i-scale:i+1].max() - current_data['low'].iloc[i-scale:i+1].min()
                net_movement = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-scale])
                if price_range > 0:
                    path_efficiency = net_movement / price_range
                    momentum_efficiency += path_efficiency / len(scales)
        
        # Volume Fractal Dimension Analysis
        volume_fractal = 0
        volume_components = []
        for scale in scales:
            if i >= scale:
                # Calculate volume clustering and burst patterns
                volume_data = current_data['volume'].iloc[i-scale:i+1]
                volume_range = volume_data.max() - volume_data.min()
                if volume_range > 0:
                    # Normalized volume variance as fractal dimension proxy
                    volume_var = volume_data.var() / (volume_range ** 2)
                    volume_components.append(volume_var)
        
        if volume_components:
            volume_fractal = np.mean(volume_components)
        
        # Price-Volume Fractal Correlation
        pv_correlation = 0
        corr_components = []
        for scale in scales:
            if i >= scale:
                # Rolling correlation between price changes and volume
                price_changes = current_data['close'].iloc[i-scale:i+1].pct_change().dropna()
                volume_changes = current_data['volume'].iloc[i-scale:i+1].pct_change().dropna()
                
                if len(price_changes) > 1 and len(volume_changes) > 1:
                    min_len = min(len(price_changes), len(volume_changes))
                    corr = np.corrcoef(price_changes.iloc[-min_len:], volume_changes.iloc[-min_len:])[0, 1]
                    if not np.isnan(corr):
                        corr_components.append(abs(corr))
        
        if corr_components:
            pv_correlation = np.mean(corr_components)
        
        # Microstructure Fractal Patterns (using OHLC and volume)
        microstructure = 0
        if i >= 10:
            # Calculate intraday price range efficiency
            recent_data = current_data.iloc[i-9:i+1]
            daily_ranges = recent_data['high'] - recent_data['low']
            price_efficiency = (recent_data['close'] - recent_data['open']).abs() / daily_ranges
            microstructure = price_efficiency.mean()
        
        # Combine all components with weights
        factor.iloc[i] = (
            0.4 * momentum_efficiency +
            0.3 * volume_fractal +
            0.2 * pv_correlation +
            0.1 * microstructure
        )
    
    # Handle initial NaN values
    factor = factor.fillna(0)
    
    return factor
