import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import entropy

def heuristics_v2(df):
    """
    Cross-Scale Entropic Momentum Resonance factor
    Combines multi-horizon entropy analysis, price-volume information coherence,
    and fractal-microstructure interactions to detect momentum regime shifts
    """
    
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # 1. Multi-Horizon Entropic State Analysis
    # Calculate Shannon entropy of price movements across different windows
    def calculate_price_entropy(price_series, window):
        """Calculate Shannon entropy of price movements"""
        returns = price_series.pct_change().dropna()
        if len(returns) < window:
            return pd.Series([np.nan] * len(price_series), index=price_series.index)
        
        entropy_vals = []
        for i in range(len(returns)):
            if i < window - 1:
                entropy_vals.append(np.nan)
            else:
                window_returns = returns.iloc[i-window+1:i+1]
                # Discretize returns into 5 bins for entropy calculation
                hist, _ = np.histogram(window_returns, bins=5, density=True)
                hist = hist[hist > 0]  # Remove zero bins for log calculation
                if len(hist) > 1:
                    ent = entropy(hist)
                    entropy_vals.append(ent)
                else:
                    entropy_vals.append(0)
        
        return pd.Series(entropy_vals, index=returns.index).reindex(price_series.index)
    
    # Calculate entropy across multiple horizons
    data['entropy_4d'] = calculate_price_entropy(data['close'], 4)
    data['entropy_7d'] = calculate_price_entropy(data['close'], 7)
    data['entropy_11d'] = calculate_price_entropy(data['close'], 11)
    
    # Compute entropic persistence using rolling autocorrelation
    def entropic_persistence(entropy_series, window=10):
        """Calculate autocorrelation of entropy values"""
        autocorr_vals = []
        for i in range(len(entropy_series)):
            if i < window - 1:
                autocorr_vals.append(np.nan)
            else:
                window_data = entropy_series.iloc[i-window+1:i+1].dropna()
                if len(window_data) >= 2:
                    autocorr = window_data.autocorr(lag=1)
                    autocorr_vals.append(autocorr if not pd.isna(autocorr) else 0)
                else:
                    autocorr_vals.append(0)
        return pd.Series(autocorr_vals, index=entropy_series.index)
    
    data['entropy_persistence'] = entropic_persistence(data['entropy_7d'])
    
    # 2. Price-Volume Information Coherence
    # Calculate mutual information proxy between price changes and volume changes
    def information_coherence(price_series, volume_series, window=10):
        """Calculate mutual information proxy using correlation and volatility"""
        price_changes = price_series.pct_change()
        volume_changes = volume_series.pct_change()
        
        coherence_vals = []
        for i in range(len(price_changes)):
            if i < window - 1:
                coherence_vals.append(np.nan)
            else:
                price_window = price_changes.iloc[i-window+1:i+1]
                volume_window = volume_changes.iloc[i-window+1:i+1]
                
                valid_mask = (~price_window.isna()) & (~volume_window.isna())
                if valid_mask.sum() >= 5:  # Minimum valid observations
                    corr = price_window[valid_mask].corr(volume_window[valid_mask])
                    # Mutual information proxy using correlation magnitude
                    mi_proxy = np.sqrt(np.abs(corr)) if not pd.isna(corr) else 0
                    coherence_vals.append(mi_proxy)
                else:
                    coherence_vals.append(0)
        
        return pd.Series(coherence_vals, index=price_series.index)
    
    data['info_coherence'] = information_coherence(data['close'], data['volume'])
    
    # 3. Fractal-Microstructure Interaction
    # Analyze high-low range compression/expansion cycles
    def range_compression_cycles(high_series, low_series, window=5):
        """Calculate normalized range compression/expansion patterns"""
        daily_range = (high_series - low_series) / high_series  # Normalized range
        
        compression_vals = []
        for i in range(len(daily_range)):
            if i < window - 1:
                compression_vals.append(np.nan)
            else:
                range_window = daily_range.iloc[i-window+1:i+1]
                # Compression score: low variance in daily ranges
                compression_score = 1 - (range_window.std() / range_window.mean()) if range_window.mean() > 0 else 0
                compression_vals.append(compression_score)
        
        return pd.Series(compression_vals, index=high_series.index)
    
    data['range_compression'] = range_compression_cycles(data['high'], data['low'])
    
    # Compute microstructure noise using intraday price oscillations
    def microstructure_noise(open_series, high_series, low_series, close_series):
        """Calculate microstructure noise from intraday price oscillations"""
        # Intraday volatility proxy using open-close relationship and range
        intraday_vol = (high_series - low_series) / open_series
        open_close_move = np.abs((close_series - open_series) / open_series)
        
        # Microstructure noise: difference between total range and open-close move
        noise = intraday_vol - open_close_move
        return noise.clip(lower=0)  # Ensure non-negative
    
    data['microstructure_noise'] = microstructure_noise(data['open'], data['high'], data['low'], data['close'])
    
    # 4. Adaptive Entropic-Regime Signals
    # Combine signals for final factor
    def calculate_final_factor(data):
        """Combine all components into final factor"""
        
        # High entropy persistence + information coherence: trend establishment
        trend_signal = data['entropy_persistence'] * data['info_coherence']
        
        # Low entropy persistence + information divergence: regime instability  
        instability_signal = (1 - np.abs(data['entropy_persistence'])) * (1 - data['info_coherence'])
        
        # Entropic regime shifts with microstructure confirmation
        entropy_gradient = data['entropy_11d'] - data['entropy_4d']  # Multi-scale entropy difference
        regime_shift_signal = np.abs(entropy_gradient) * data['microstructure_noise']
        
        # Fractal-microstructure convergence
        fractal_signal = data['range_compression'] * data['microstructure_noise']
        
        # Final factor: weighted combination of regime signals
        final_factor = (
            0.4 * trend_signal +           # Trend establishment
            -0.3 * instability_signal +    # Regime instability (negative signal)
            0.5 * regime_shift_signal +    # Momentum ignition
            0.2 * fractal_signal           # Volatility clustering
        )
        
        return final_factor
    
    # Calculate final factor
    factor = calculate_final_factor(data)
    
    # Normalize and clean the factor
    factor = (factor - factor.rolling(window=20, min_periods=10).mean()) / factor.rolling(window=20, min_periods=10).std()
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    return factor
