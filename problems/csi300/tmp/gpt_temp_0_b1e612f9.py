import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import entropy

def heuristics_v2(df):
    """
    Hierarchical Fractal Efficiency Momentum with Volume-Entropy Informed Reversal
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # 1. Hierarchical Fractal Efficiency Momentum
    # Calculate price path efficiency using fractal dimension approximation
    def fractal_efficiency(high, low, close, window=20):
        """Approximate fractal dimension using price range efficiency"""
        price_range = high - low
        price_movement = close.diff().abs()
        
        # Efficiency ratio: net movement vs total range
        efficiency = price_movement.rolling(window=window).sum() / price_range.rolling(window=window).sum()
        efficiency = efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return efficiency
    
    # Multi-timeframe momentum congruence
    def momentum_congruence(close, short_window=10, medium_window=20, long_window=50):
        """Measure momentum alignment across timeframes"""
        mom_short = close.pct_change(short_window)
        mom_medium = close.pct_change(medium_window)
        mom_long = close.pct_change(long_window)
        
        # Congruence score: product of normalized momentums
        congruence = (mom_short * mom_medium * mom_long).abs()
        congruence = congruence.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return congruence
    
    # Efficiency persistence across regimes
    def efficiency_persistence(efficiency, regime_window=30):
        """Measure how efficiency persists across market regimes"""
        efficiency_std = efficiency.rolling(window=regime_window).std()
        efficiency_mean = efficiency.rolling(window=regime_window).mean()
        
        # Persistence score: inverse of efficiency volatility
        persistence = 1 / (efficiency_std + 1e-8)
        persistence = persistence.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return persistence
    
    # 2. Volume-Entropy Informed Reversal
    def volume_entropy(volume, window=20, bins=10):
        """Compute information entropy of volume distribution"""
        entropy_values = []
        for i in range(len(volume)):
            if i < window:
                entropy_values.append(0)
                continue
                
            window_volumes = volume.iloc[i-window:i]
            # Create histogram and calculate entropy
            hist, _ = np.histogram(window_volumes, bins=bins)
            prob = hist / hist.sum() if hist.sum() > 0 else np.ones_like(hist) / len(hist)
            ent = entropy(prob)
            entropy_values.append(ent)
        
        return pd.Series(entropy_values, index=volume.index)
    
    def price_reversal_strength(close, volume_entropy_series, window=10):
        """Detect price reversal strength using entropy extremes"""
        price_returns = close.pct_change()
        
        # Identify entropy extremes (high entropy suggests disorder/reversal potential)
        entropy_high = volume_entropy_series.rolling(window=window).quantile(0.8)
        entropy_low = volume_entropy_series.rolling(window=window).quantile(0.2)
        
        # Reversal strength when high entropy coincides with price movement
        reversal_strength = np.where(
            volume_entropy_series > entropy_high,
            -price_returns,  # Negative for mean reversion
            0
        )
        
        return pd.Series(reversal_strength, index=close.index)
    
    def volume_clustering_asymmetry(volume, window=20):
        """Measure asymmetry in volume clustering"""
        volume_ma = volume.rolling(window=window).mean()
        volume_std = volume.rolling(window=window).std()
        
        # Z-score of volume
        volume_z = (volume - volume_ma) / (volume_std + 1e-8)
        
        # Asymmetry: difference between positive and negative z-score clustering
        positive_cluster = volume_z[volume_z > 0].rolling(window=5).mean()
        negative_cluster = volume_z[volume_z < 0].rolling(window=5).mean()
        
        asymmetry = (positive_cluster - negative_cluster).abs()
        asymmetry = asymmetry.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return asymmetry
    
    # Calculate components
    efficiency = fractal_efficiency(data['high'], data['low'], data['close'])
    congruence = momentum_congruence(data['close'])
    persistence = efficiency_persistence(efficiency)
    
    # Hierarchical Fractal Efficiency Momentum
    hfem = efficiency * congruence * persistence
    
    # Volume-Entropy components
    vol_entropy = volume_entropy(data['volume'])
    reversal_str = price_reversal_strength(data['close'], vol_entropy)
    vol_asymmetry = volume_clustering_asymmetry(data['volume'])
    
    # Volume-Entropy Informed Reversal
    veir = reversal_str * vol_asymmetry
    
    # Combine factors with appropriate scaling
    # Normalize both factors before combination
    hfem_normalized = (hfem - hfem.rolling(window=50).mean()) / (hfem.rolling(window=50).std() + 1e-8)
    veir_normalized = (veir - veir.rolling(window=50).mean()) / (veir.rolling(window=50).std() + 1e-8)
    
    # Final factor: weighted combination
    final_factor = 0.6 * hfem_normalized + 0.4 * veir_normalized
    
    # Clean infinite values and NaNs
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return final_factor
