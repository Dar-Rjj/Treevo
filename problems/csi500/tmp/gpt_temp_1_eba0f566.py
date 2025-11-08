import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import entropy

def heuristics_v2(df):
    """
    Multi-Scale Price-Volume Entropy Divergence factor
    """
    data = df.copy()
    
    # Calculate daily returns for entropy analysis
    returns = data['close'].pct_change()
    
    # Multi-Horizon Entropy Analysis
    def calculate_entropy(series, window):
        """Calculate Shannon entropy of returns over specified window"""
        if len(series) < window:
            return pd.Series([np.nan] * len(series), index=series.index)
        
        entropy_vals = []
        for i in range(len(series)):
            if i < window - 1:
                entropy_vals.append(np.nan)
            else:
                window_data = series.iloc[i-window+1:i+1].dropna()
                if len(window_data) < 2:
                    entropy_vals.append(np.nan)
                else:
                    # Discretize returns into bins for entropy calculation
                    hist, _ = np.histogram(window_data, bins=10, density=True)
                    hist = hist[hist > 0]  # Remove zero bins
                    if len(hist) > 1:
                        entropy_vals.append(entropy(hist))
                    else:
                        entropy_vals.append(np.nan)
        return pd.Series(entropy_vals, index=series.index)
    
    # Price entropy at different horizons
    entropy_3d = calculate_entropy(returns, 3)
    entropy_8d = calculate_entropy(returns, 8)
    entropy_21d = calculate_entropy(returns, 21)
    
    # Entropy divergence detection
    short_medium_div = entropy_3d - entropy_8d
    medium_long_div = entropy_8d - entropy_21d
    
    # Volume Fractal Dynamics
    volume = data['volume']
    
    # Volume concentration
    volume_5d_sum = volume.rolling(window=5, min_periods=1).sum()
    volume_concentration = volume / volume_5d_sum
    
    # Volume skewness and kurtosis
    volume_skewness = volume.rolling(window=10, min_periods=5).skew()
    volume_kurtosis = volume.rolling(window=10, min_periods=5).kurt()
    
    # Multi-scale volume ratios
    volume_ratio_1d = volume / volume.shift(1)
    volume_ratio_3d = volume / volume.shift(3)
    volume_ratio_8d = volume / volume.shift(8)
    
    # Volume fractal dimension (simplified as complexity measure)
    def volume_fractal_dimension(vol_series, window=10):
        """Calculate simplified fractal dimension of volume pattern"""
        fractal_dims = []
        for i in range(len(vol_series)):
            if i < window - 1:
                fractal_dims.append(np.nan)
            else:
                window_data = vol_series.iloc[i-window+1:i+1]
                if len(window_data.dropna()) < window:
                    fractal_dims.append(np.nan)
                else:
                    # Simplified fractal dimension using range-to-std ratio
                    data_range = window_data.max() - window_data.min()
                    data_std = window_data.std()
                    if data_std > 0:
                        fractal_dims.append(data_range / data_std)
                    else:
                        fractal_dims.append(np.nan)
        return pd.Series(fractal_dims, index=vol_series.index)
    
    volume_fractal = volume_fractal_dimension(volume, 10)
    
    # Price-Volume Information Asymmetry
    price_change = abs(data['close'] - data['close'].shift(1))
    price_volume_ratio = price_change / volume
    volume_weighted_info = volume * price_change
    
    # Directional information asymmetry (simplified using amount/volume ratio)
    directional_asymmetry = data['amount'] / (volume * data['close'])
    
    # Market Microstructure Regimes
    # Volume-based liquidity
    volume_percentile = volume.rolling(window=5, min_periods=3).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Spread efficiency
    spread_efficiency = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    
    # Information regime detection
    def news_impact_probability(vol, price_chg, window=5):
        """Detect volume spikes relative to price movement"""
        impact_prob = []
        for i in range(len(vol)):
            if i < window - 1:
                impact_prob.append(np.nan)
            else:
                vol_window = vol.iloc[i-window+1:i+1]
                price_window = price_chg.iloc[i-window+1:i+1]
                current_vol_z = (vol.iloc[i] - vol_window.mean()) / vol_window.std() if vol_window.std() > 0 else 0
                current_price_z = (price_chg.iloc[i] - price_window.mean()) / price_window.std() if price_window.std() > 0 else 0
                # High impact when high volume but low price movement
                impact_prob.append(current_vol_z - current_price_z)
        return pd.Series(impact_prob, index=vol.index)
    
    news_impact = news_impact_probability(volume, price_change, 5)
    
    # Adaptive Entropy Integration
    # Liquidity regime classification
    liquidity_regime = volume_percentile.rolling(window=5, min_periods=3).mean()
    
    # Regime-dependent entropy weighting
    def regime_weighting(liquidity, price_entropy, volume_entropy):
        """Apply regime-specific weighting"""
        # High liquidity: emphasize price entropy
        # Low liquidity: emphasize volume entropy
        regime_factor = np.tanh(liquidity)  # Map to [-1, 1]
        weight_price = (1 + regime_factor) / 2  # [0, 1]
        weight_volume = 1 - weight_price
        return weight_price * price_entropy + weight_volume * volume_entropy
    
    # Create volume entropy proxy using volume concentration and fractal dimension
    volume_entropy_proxy = volume_concentration * volume_fractal
    
    # Apply regime weighting
    regime_adjusted_entropy = regime_weighting(
        liquidity_regime, 
        (entropy_3d + entropy_8d + entropy_21d) / 3,  # Average price entropy
        volume_entropy_proxy
    )
    
    # Base factor: Price Entropy Divergence × Volume Fractal Dimension
    price_entropy_divergence = (short_medium_div + medium_long_div) / 2
    base_factor = price_entropy_divergence * volume_fractal
    
    # Integrated alpha generation with regime adjustment
    integrated_alpha = base_factor * regime_adjusted_entropy
    
    # Local scaling using 10-day rolling standard deviation
    alpha_scaled = integrated_alpha / integrated_alpha.rolling(window=10, min_periods=5).std()
    
    # Final factor with NaN handling
    final_factor = alpha_scaled.fillna(0)
    
    return final_factor
