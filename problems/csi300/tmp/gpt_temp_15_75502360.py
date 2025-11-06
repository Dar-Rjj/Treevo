import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit

def heuristics_v2(df):
    """
    Nonlinear Temporal Pattern Recognition Framework for alpha generation
    Combines fractal analysis, chaos theory, and multi-scale pattern recognition
    """
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Ensure sufficient data for calculations
    min_periods = 50
    if len(df) < min_periods:
        return alpha
    
    # Price series for analysis
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    volumes = df['volume'].values
    
    # 1. Fractal Market Structure Analysis
    def compute_hurst_exponent(series, max_lag=20):
        """Compute Hurst exponent using rescaled range method"""
        if len(series) < max_lag * 2:
            return 0.5
        
        lags = range(2, min(max_lag, len(series)//2))
        rs_values = []
        
        for lag in lags:
            # Split series into non-overlapping windows
            n_windows = len(series) // lag
            if n_windows < 2:
                continue
                
            rs_window = []
            for i in range(n_windows):
                segment = series[i*lag:(i+1)*lag]
                if len(segment) < 2:
                    continue
                    
                # Calculate mean-adjusted series
                mean_segment = np.mean(segment)
                adjusted = segment - mean_segment
                cumulative = np.cumsum(adjusted)
                
                # Range and standard deviation
                R = np.max(cumulative) - np.min(cumulative)
                S = np.std(segment, ddof=1)
                
                if S > 0:
                    rs_window.append(R / S)
            
            if rs_window:
                rs_values.append(np.mean(rs_window))
        
        if len(rs_values) < 2:
            return 0.5
            
        # Linear regression in log-log space
        lags_array = np.array(lags[:len(rs_values)])
        try:
            slope, _, _, _, _ = linregress(np.log(lags_array), np.log(rs_values))
            return slope
        except:
            return 0.5
    
    # 2. Chaos Theory Market Dynamics
    def compute_lyapunov_exponent(returns, embedding_dim=3, tau=1):
        """Estimate largest Lyapunov exponent from return series"""
        if len(returns) < embedding_dim * 10:
            return 0
        
        # Phase space reconstruction
        n_vectors = len(returns) - (embedding_dim - 1) * tau
        if n_vectors < 10:
            return 0
            
        # Create embedded vectors
        embedded = np.zeros((n_vectors, embedding_dim))
        for i in range(n_vectors):
            for j in range(embedding_dim):
                embedded[i, j] = returns[i + j * tau]
        
        # Find nearest neighbors and track divergence
        divergences = []
        for i in range(min(100, n_vectors-1)):
            # Find nearest neighbor (excluding immediate neighbors)
            distances = np.linalg.norm(embedded - embedded[i], axis=1)
            distances[max(0, i-5):min(n_vectors, i+5)] = np.inf  # Exclude nearby points
            
            if np.all(np.isinf(distances)):
                continue
                
            nn_idx = np.argmin(distances)
            initial_dist = distances[nn_idx]
            
            # Track evolution
            evolution_steps = min(10, n_vectors - max(i, nn_idx) - 1)
            if evolution_steps > 0:
                final_dist = np.linalg.norm(embedded[i + evolution_steps] - 
                                          embedded[nn_idx + evolution_steps])
                if initial_dist > 0 and final_dist > 0:
                    divergences.append(np.log(final_dist / initial_dist) / evolution_steps)
        
        return np.mean(divergences) if divergences else 0
    
    # 3. Wavelet-Based Multi-Scale Analysis
    def multi_scale_volatility(prices, scales=[5, 10, 20]):
        """Compute volatility at different time scales"""
        volatilities = []
        for scale in scales:
            if len(prices) >= scale:
                returns = np.diff(np.log(prices[-scale:]))
                if len(returns) > 1:
                    vol = np.std(returns)
                    volatilities.append(vol)
                else:
                    volatilities.append(0)
            else:
                volatilities.append(0)
        
        # Measure volatility term structure slope
        if len(volatilities) >= 2:
            try:
                slope, _, _, _, _ = linregress(range(len(volatilities)), volatilities)
                return slope
            except:
                return 0
        return 0
    
    # 4. Information Geometry of Order Flow
    def volume_price_fisher_info(volumes, prices, window=20):
        """Compute Fisher information from volume-price relationship"""
        if len(volumes) < window or len(prices) < window:
            return 0
        
        recent_volumes = volumes[-window:]
        recent_prices = prices[-window:]
        
        # Normalize
        norm_volumes = (recent_volumes - np.mean(recent_volumes)) / (np.std(recent_volumes) + 1e-8)
        norm_prices = (recent_prices - np.mean(recent_prices)) / (np.std(recent_prices) + 1e-8)
        
        # Joint distribution moments
        covariance = np.cov(norm_volumes, norm_prices)[0, 1]
        
        # Fisher information approximation
        fisher_info = covariance ** 2 / (1 - covariance ** 2 + 1e-8)
        return fisher_info
    
    # Calculate features for each day
    for i in range(min_periods, len(df)):
        # Use only past data up to current day
        current_data = df.iloc[:i+1]
        
        if len(current_data) < min_periods:
            alpha.iloc[i] = 0
            continue
        
        # Extract recent series
        recent_close = current_data['close'].values
        recent_high = current_data['high'].values
        recent_low = current_data['low'].values
        recent_volume = current_data['volume'].values
        
        # Calculate returns for chaos analysis
        returns = np.diff(np.log(recent_close))
        if len(returns) < 10:
            alpha.iloc[i] = 0
            continue
        
        # Compute individual components
        hurst_exp = compute_hurst_exponent(recent_close[-100:])  # Last 100 days
        lyapunov_exp = compute_lyapunov_exponent(returns[-50:])  # Last 50 returns
        vol_term_structure = multi_scale_volatility(recent_close[-30:])  # Last 30 days
        fisher_info = volume_price_fisher_info(recent_volume[-20:], recent_close[-20:])
        
        # Additional pattern recognition features
        # Price momentum persistence
        recent_momentum = np.mean(np.diff(recent_close[-10:])) / (np.std(np.diff(recent_close[-10:])) + 1e-8)
        
        # Volume-price correlation
        volume_corr = np.corrcoef(recent_volume[-20:], recent_close[-20:])[0, 1]
        if np.isnan(volume_corr):
            volume_corr = 0
        
        # Range efficiency (true range vs close-to-close movement)
        true_ranges = []
        for j in range(1, min(21, len(recent_high))):
            tr = max(recent_high[-j] - recent_low[-j], 
                    abs(recent_high[-j] - recent_close[-(j+1)]),
                    abs(recent_low[-j] - recent_close[-(j+1)]))
            true_ranges.append(tr)
        
        if true_ranges:
            range_efficiency = np.mean(np.abs(np.diff(recent_close[-20:]))) / (np.mean(true_ranges) + 1e-8)
        else:
            range_efficiency = 0
        
        # Combine features into alpha factor
        # Weights determined by empirical significance
        alpha_value = (
            0.3 * hurst_exp +                    # Fractal structure
            0.25 * lyapunov_exp +                # Chaos dynamics
            0.2 * vol_term_structure +           # Multi-scale volatility
            0.15 * fisher_info +                 # Information geometry
            0.05 * recent_momentum +             # Short-term momentum
            0.03 * volume_corr +                 # Volume-price relationship
            0.02 * range_efficiency              # Market efficiency
        )
        
        alpha.iloc[i] = alpha_value
    
    # Normalize the alpha series
    if len(alpha.dropna()) > 0:
        alpha = (alpha - alpha.mean()) / (alpha.std() + 1e-8)
    
    return alpha
