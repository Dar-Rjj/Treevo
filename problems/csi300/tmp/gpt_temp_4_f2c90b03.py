import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence with Fractal Market Structure Analysis
    Generates alpha factor combining fractal dimension analysis with price-volume divergence
    """
    
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Helper function for fractal dimension calculation using box-counting method
    def calculate_fractal_dimension(series, window):
        """Calculate fractal dimension using simplified box-counting method"""
        fractals = []
        for i in range(len(series)):
            if i < window - 1:
                fractals.append(np.nan)
                continue
                
            window_data = series.iloc[i-window+1:i+1]
            if len(window_data) < window:
                fractals.append(np.nan)
                continue
                
            # Normalize the series to [0,1] range
            normalized = (window_data - window_data.min()) / (window_data.max() - window_data.min() + 1e-12)
            
            # Simplified box-counting: count directional changes
            changes = np.abs(np.diff(normalized.values))
            total_path = np.sum(changes)
            max_possible = np.sqrt(2)  # Maximum possible path in normalized space
            
            # Fractal dimension approximation
            if total_path > 0:
                fractal_dim = 1 + (np.log(total_path + 1) / np.log(window))
                fractals.append(fractal_dim)
            else:
                fractals.append(1.0)
                
        return pd.Series(fractals, index=series.index)
    
    # Calculate price fractal dimensions
    data['fractal_5d'] = calculate_fractal_dimension(data['close'], 5)
    data['fractal_20d'] = calculate_fractal_dimension(data['close'], 20)
    data['fractal_ratio'] = data['fractal_5d'] / data['fractal_20d']
    
    # Calculate volume fractal dimension
    data['volume_fractal_10d'] = calculate_fractal_dimension(data['volume'], 10)
    
    # Price-volume divergence calculations
    # 1. Directional divergence
    data['price_change'] = data['close'].pct_change(5)
    data['volume_change'] = data['volume'].pct_change(5)
    
    # Directional divergence signals
    data['divergence_directional'] = 0
    # Price up, volume down
    mask_up_down = (data['price_change'] > 0.02) & (data['volume_change'] < -0.1)
    data.loc[mask_up_down, 'divergence_directional'] = -1
    # Price down, volume up
    mask_down_up = (data['price_change'] < -0.02) & (data['volume_change'] > 0.1)
    data.loc[mask_down_up, 'divergence_directional'] = 1
    
    # 2. Magnitude divergence
    data['price_magnitude'] = (data['high'] - data['low']) / data['close']
    data['volume_magnitude'] = data['volume'] / data['volume'].rolling(20, min_periods=1).mean()
    
    data['magnitude_divergence'] = (data['price_magnitude'] - data['volume_magnitude'].rolling(5).mean()) / \
                                  (data['price_magnitude'].rolling(10).std() + 1e-12)
    
    # Market regime classification using fractal properties
    data['fractal_regime'] = 0  # 0: ranging, 1: trending, 2: transition
    
    # High fractal dimension = trending markets
    trending_mask = (data['fractal_20d'] > data['fractal_20d'].rolling(50).quantile(0.7))
    data.loc[trending_mask, 'fractal_regime'] = 1
    
    # Low fractal dimension = ranging markets
    ranging_mask = (data['fractal_20d'] < data['fractal_20d'].rolling(50).quantile(0.3))
    data.loc[ranging_mask, 'fractal_regime'] = 0
    
    # Changing fractal dimension = regime transitions
    fractal_change = data['fractal_20d'].diff(5).abs()
    transition_mask = (fractal_change > fractal_change.rolling(20).quantile(0.7))
    data.loc[transition_mask, 'fractal_regime'] = 2
    
    # Volume fractal regime correlation
    data['volume_fractal_change'] = data['volume_fractal_10d'].diff(3)
    data['volume_clustering'] = data['volume'].rolling(5).std() / (data['volume'].rolling(20).std() + 1e-12)
    
    # Fractal-weighted divergence amplification
    # Higher weights during regime transitions
    regime_weights = data['fractal_regime'].map({0: 0.5, 1: 0.8, 2: 1.2})
    
    # Weight by fractal dimension changes
    fractal_change_weight = 1 + data['fractal_ratio'].diff(3).abs() * 2
    
    # Multi-timeframe divergence integration
    # Short-term divergence (5-day)
    data['divergence_5d'] = data['divergence_directional'] * data['magnitude_divergence'].rolling(5).mean()
    
    # Medium-term divergence (20-day)
    data['divergence_20d'] = data['divergence_directional'].rolling(10).mean() * \
                            data['magnitude_divergence'].rolling(20).mean()
    
    # Cross-timeframe divergence confirmation
    data['divergence_confirmation'] = np.sign(data['divergence_5d']) * np.sign(data['divergence_20d'])
    
    # Final alpha factor synthesis
    # Primary factor: Fractal-regime weighted price-volume divergence score
    primary_factor = (data['divergence_5d'] * regime_weights * fractal_change_weight * 
                     (1 + data['volume_clustering']))
    
    # Confirmation layer: Multi-scale fractal alignment
    fractal_alignment = data['fractal_ratio'].rolling(10).std()
    volume_fractal_alignment = data['volume_fractal_10d'].rolling(10).std()
    
    confirmation_signal = (1 - fractal_alignment) * (1 - volume_fractal_alignment)
    
    # Combine primary factor with confirmation signals
    alpha_factor = primary_factor * confirmation_signal * data['divergence_confirmation']
    
    # Normalize the final factor
    alpha_normalized = (alpha_factor - alpha_factor.rolling(50).mean()) / \
                      (alpha_factor.rolling(50).std() + 1e-12)
    
    return alpha_normalized
