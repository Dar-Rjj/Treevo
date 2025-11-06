import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate for each day using only past data
    for i in range(10, len(data)):
        current_data = data.iloc[:i+1].copy()
        
        # Asymmetric Volatility Momentum Component
        if i >= 10:
            # Get recent returns (t-10 to t)
            recent_returns = current_data['returns'].iloc[-11:]
            
            # Upside volatility (positive returns only)
            pos_returns = recent_returns[recent_returns > 0]
            upside_vol = pos_returns.std() if len(pos_returns) > 1 else 0
            
            # Downside volatility (negative returns only)
            neg_returns = recent_returns[recent_returns < 0]
            downside_vol = neg_returns.std() if len(neg_returns) > 1 else 0
            
            # Avoid division by zero
            vol_ratio = upside_vol / downside_vol if downside_vol > 0 else 1
            
            # Price momentum using linear regression slope
            if i >= 5:
                momentum_data = current_data['close'].iloc[-6:]
                x = np.arange(len(momentum_data))
                slope, _, _, _, _ = stats.linregress(x, momentum_data)
                momentum = slope
            else:
                momentum = 0
                
            asym_vol_momentum = momentum * vol_ratio
        else:
            asym_vol_momentum = 0
        
        # Volume-Price Divergence Component
        if i >= 8:
            # Price oscillator (short MA - long MA)
            price_short = current_data['close'].iloc[-5:].mean()
            price_long = current_data['close'].iloc[-9:].mean()
            price_osc = price_short - price_long
            
            # Volume oscillator (short MA - long MA)
            vol_short = current_data['volume'].iloc[-5:].mean()
            vol_long = current_data['volume'].iloc[-9:].mean()
            vol_osc = vol_short - vol_long
            
            # Divergence angle using correlation
            price_window = current_data['close'].iloc[-8:]
            vol_window = current_data['volume'].iloc[-8:]
            
            if len(price_window) > 1 and len(vol_window) > 1:
                correlation = np.corrcoef(price_window, vol_window)[0,1]
                if not np.isnan(correlation):
                    divergence_angle = np.arccos(np.clip(correlation, -1, 1))
                else:
                    divergence_angle = np.pi/2
            else:
                divergence_angle = np.pi/2
            
            # Activity intensity
            price_change = abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-2])
            vol_change_pct = (current_data['volume'].iloc[-1] - current_data['volume'].iloc[-2]) / current_data['volume'].iloc[-2] if current_data['volume'].iloc[-2] > 0 else 0
            
            activity_intensity = price_change * (1 + abs(vol_change_pct))
            volume_price_divergence = divergence_angle * activity_intensity
        else:
            volume_price_divergence = 0
        
        # Liquidity Clustering Component
        if i >= 15:
            # Volume spikes detection
            volume_window = current_data['volume'].iloc[-16:]
            volume_median = volume_window.rolling(window=5, min_periods=1).median().iloc[-1]
            
            # Identify volume spikes
            spike_threshold = 2 * volume_median
            spike_days = volume_window > spike_threshold
            
            if spike_days.any():
                # Get close prices at spike days
                spike_prices = current_data['close'].iloc[-16:][spike_days]
                
                if len(spike_prices) > 0:
                    # Calculate cluster boundaries
                    cluster_min = spike_prices.min()
                    cluster_max = spike_prices.max()
                    current_close = current_data['close'].iloc[-1]
                    
                    # Distance to nearest cluster boundary
                    if current_close < cluster_min:
                        distance = cluster_min - current_close
                    elif current_close > cluster_max:
                        distance = current_close - cluster_max
                    else:
                        distance = 0
                    
                    # Volume confirmation
                    recent_volumes = current_data['volume'].iloc[-10:]
                    current_volume = current_data['volume'].iloc[-1]
                    volume_percentile = (recent_volumes < current_volume).sum() / len(recent_volumes)
                    
                    liquidity_clustering = (1 - distance / (cluster_max - cluster_min)) * volume_percentile if (cluster_max - cluster_min) > 0 else 0
                else:
                    liquidity_clustering = 0
            else:
                liquidity_clustering = 0
        else:
            liquidity_clustering = 0
        
        # Regime-Switching Component
        if i >= 10:
            # Calculate autocorrelation for trend persistence
            returns_window = current_data['returns'].iloc[-11:]
            if len(returns_window) > 2:
                autocorr = returns_window.autocorr()
                if np.isnan(autocorr):
                    autocorr = 0
            else:
                autocorr = 0
            
            # Simple regime classification
            regime_strength = abs(autocorr)
            
            # Calculate overextension from median
            price_window = current_data['close'].iloc[-21:]
            price_median = price_window.median()
            current_price = current_data['close'].iloc[-1]
            
            # Regime-appropriate lookback
            lookback = 10 if regime_strength > 0.3 else 20  # shorter lookback in trending regimes
            
            deviation = (current_price - price_median) / price_median if price_median > 0 else 0
            
            # Apply regime-dependent reversion
            if regime_strength > 0.3:  # Trending regime
                regime_reversion = deviation * 0.5  # Weak reversion
            else:  # Mean-reverting regime
                regime_reversion = deviation * 2.0  # Strong reversion
                
            regime_switching = regime_reversion * (1 - regime_strength)
        else:
            regime_switching = 0
        
        # Price-Volume Harmony Component
        if i >= 3:
            # Price velocity (acceleration)
            price_window = current_data['close'].iloc[-4:]
            if len(price_window) > 2:
                price_roc1 = (price_window.iloc[-1] - price_window.iloc[-2]) / price_window.iloc[-2] if price_window.iloc[-2] > 0 else 0
                price_roc2 = (price_window.iloc[-2] - price_window.iloc[-3]) / price_window.iloc[-3] if price_window.iloc[-3] > 0 else 0
                price_acceleration = price_roc1 - price_roc2
            else:
                price_acceleration = 0
            
            # Volume velocity (acceleration)
            volume_window = current_data['volume'].iloc[-4:]
            if len(volume_window) > 2:
                volume_roc1 = (volume_window.iloc[-1] - volume_window.iloc[-2]) / volume_window.iloc[-2] if volume_window.iloc[-2] > 0 else 0
                volume_roc2 = (volume_window.iloc[-2] - volume_window.iloc[-3]) / volume_window.iloc[-3] if volume_window.iloc[-3] > 0 else 0
                volume_acceleration = volume_roc1 - volume_roc2
            else:
                volume_acceleration = 0
            
            # Phase alignment (angular difference)
            if price_acceleration != 0 and volume_acceleration != 0:
                # Calculate angle between vectors
                dot_product = price_acceleration * volume_acceleration
                magnitude_product = np.sqrt(price_acceleration**2 + 1e-8) * np.sqrt(volume_acceleration**2 + 1e-8)
                cosine_similarity = dot_product / magnitude_product
                phase_alignment = np.arccos(np.clip(cosine_similarity, -1, 1))
            else:
                phase_alignment = np.pi/2
            
            # Magnitude consistency
            price_magnitude = abs(price_acceleration)
            volume_magnitude = abs(volume_acceleration)
            
            if volume_magnitude > 0:
                magnitude_ratio = price_magnitude / volume_magnitude
                # Logarithmic scaling to handle extremes
                magnitude_consistency = 1 / (1 + abs(np.log(magnitude_ratio + 1e-8)))
            else:
                magnitude_consistency = 0.5
            
            price_volume_harmony = (np.pi/2 - phase_alignment) * magnitude_consistency
        else:
            price_volume_harmony = 0
        
        # Combine all components with equal weights
        factor.iloc[i] = (
            asym_vol_momentum + 
            volume_price_divergence + 
            liquidity_clustering + 
            regime_switching + 
            price_volume_harmony
        )
    
    # Fill early values with 0
    factor = factor.fillna(0)
    
    return factor
