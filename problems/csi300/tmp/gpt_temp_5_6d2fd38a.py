import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Microstructure Alpha - Combines multi-scale fractal dynamics with 
    temporal asymmetry and phase relationships for predictive signals.
    """
    # Price-Volume Fractal Dynamics
    # Multi-Scale Volume Fractal Ratio
    def fractal_dimension_volume(series, window):
        """Calculate fractal dimension using Higuchi method for volume series"""
        if len(series) < window:
            return pd.Series([np.nan] * len(series), index=series.index)
        
        result = []
        for i in range(len(series)):
            if i < window - 1:
                result.append(np.nan)
                continue
                
            segment = series.iloc[i-window+1:i+1]
            if segment.isna().any():
                result.append(np.nan)
                continue
                
            # Higuchi fractal dimension calculation
            L = []
            for k in range(1, min(6, window//2)):
                Lk = 0
                for m in range(k):
                    idx = np.arange(m, len(segment), k)
                    if len(idx) < 2:
                        continue
                    segment_k = segment.iloc[idx]
                    Lkm = np.sum(np.abs(np.diff(segment_k.values))) * (len(segment) - 1) / (len(idx) * k)
                    Lk += Lkm
                L.append(np.log(Lk / k))
            
            if len(L) > 1:
                x = np.log(np.arange(1, len(L)+1))
                slope = np.polyfit(x, L, 1)[0]
                result.append(-slope)
            else:
                result.append(np.nan)
        
        return pd.Series(result, index=series.index)
    
    # Calculate volume fractal dimensions
    fd_volume_3d = fractal_dimension_volume(df['volume'], 3)
    fd_volume_10d = fractal_dimension_volume(df['volume'], 10)
    volume_fractal_ratio = fd_volume_3d / fd_volume_10d
    
    # Price Range Fractal Compression
    def fractal_dimension_range(df, window):
        """Calculate fractal dimension for price range series"""
        price_range = (df['high'] - df['low']) / df['close']
        return fractal_dimension_volume(price_range, window)
    
    fd_range_5d = fractal_dimension_range(df, 5)
    fd_range_20d = fractal_dimension_range(df, 20)
    range_fractal_compression = fd_range_5d - fd_range_20d
    
    # Fractal Dimension Acceleration
    delta_fd_volume_3d = fd_volume_3d.diff(1)
    
    # Temporal Asymmetry Patterns
    # Forward-Backward Price Path Divergence
    def forward_backward_divergence(df, window=5):
        """Calculate divergence between forward and backward price paths"""
        forward_returns = df['close'].pct_change(window).shift(-window)
        backward_returns = df['close'].pct_change(window)
        
        # Use only historical data - simulate forward path using current information
        current_trend = df['close'].rolling(window=window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
        )
        
        expected_forward = current_trend * window / df['close']
        actual_backward = backward_returns
        
        divergence = expected_forward - actual_backward
        return divergence
    
    fb_divergence = forward_backward_divergence(df)
    
    # Volume Concentration in Price Moves
    def volume_concentration(df, window=10):
        """Measure volume concentration during significant price moves"""
        price_change = df['close'].pct_change()
        significant_moves = price_change.abs() > price_change.rolling(window=20).std()
        
        concentrated_volume = df['volume'].rolling(window=window).apply(
            lambda x: np.sum(x[significant_moves.iloc[-len(x):].values]) / np.sum(x) 
            if np.sum(x) > 0 else 0, raw=False
        )
        return concentrated_volume
    
    vol_concentration = volume_concentration(df)
    
    # Order Flow Persistence Decay
    def order_flow_persistence(df, decay_window=10):
        """Calculate persistence decay in order flow patterns"""
        price_volume_corr = df['close'].rolling(window=5).corr(df['volume'])
        persistence = price_volume_corr.ewm(span=decay_window).mean()
        decay_rate = persistence.diff(3)  # 3-day change in persistence
        return decay_rate
    
    order_persistence = order_flow_persistence(df)
    
    # Multi-Frequency Phase Relationships
    # Price-Volume Phase Coherence
    def phase_coherence(df, short_window=5, long_window=20):
        """Calculate phase coherence between price and volume oscillations"""
        price_ma_short = df['close'].rolling(window=short_window).mean()
        price_ma_long = df['close'].rolling(window=long_window).mean()
        volume_ma_short = df['volume'].rolling(window=short_window).mean()
        volume_ma_long = df['volume'].rolling(window=long_window).mean()
        
        price_phase = np.arctan2(price_ma_short - price_ma_long, price_ma_long)
        volume_phase = np.arctan2(volume_ma_short - volume_ma_long, volume_ma_long)
        
        phase_diff = np.sin(price_phase - volume_phase)
        coherence = phase_diff.rolling(window=10).std()
        return coherence
    
    phase_coherence_signal = phase_coherence(df)
    
    # Phase Lead-Lag Detection
    def phase_lead_lag(df, lag_window=5):
        """Detect phase lead-lag relationships"""
        price_returns = df['close'].pct_change()
        volume_returns = df['volume'].pct_change()
        
        # Cross-correlation at different lags
        lead_lag_scores = []
        for i in range(len(df)):
            if i < lag_window:
                lead_lag_scores.append(np.nan)
                continue
            
            price_segment = price_returns.iloc[i-lag_window:i]
            volume_segment = volume_returns.iloc[i-lag_window:i]
            
            if price_segment.isna().any() or volume_segment.isna().any():
                lead_lag_scores.append(np.nan)
                continue
            
            # Calculate correlation at different lags
            max_corr = 0
            best_lag = 0
            for lag in range(-2, 3):
                if lag < 0:
                    corr = np.corrcoef(price_segment.iloc[-lag:], 
                                     volume_segment.iloc[:lag])[0,1]
                elif lag > 0:
                    corr = np.corrcoef(price_segment.iloc[:-lag], 
                                     volume_segment.iloc[lag:])[0,1]
                else:
                    corr = np.corrcoef(price_segment, volume_segment)[0,1]
                
                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_lag = lag
            
            lead_lag_scores.append(best_lag * max_corr)
        
        return pd.Series(lead_lag_scores, index=df.index)
    
    lead_lag = phase_lead_lag(df)
    
    # Nonlinear Dynamics Measures
    # Price-Volume Lyapunov Divergence
    def lyapunov_divergence(df, window=20):
        """Estimate Lyapunov-like divergence in price-volume space"""
        price_norm = (df['close'] - df['close'].rolling(window=window).mean()) / df['close'].rolling(window=window).std()
        volume_norm = (df['volume'] - df['volume'].rolling(window=window).mean()) / df['volume'].rolling(window=window).std()
        
        # Calculate local divergence rates
        divergence = []
        for i in range(len(df)):
            if i < window:
                divergence.append(np.nan)
                continue
            
            p_segment = price_norm.iloc[i-window:i]
            v_segment = volume_norm.iloc[i-window:i]
            
            if p_segment.isna().any() or v_segment.isna().any():
                divergence.append(np.nan)
                continue
            
            # Simple divergence measure
            p_grad = np.gradient(p_segment)
            v_grad = np.gradient(v_segment)
            local_div = np.mean(np.abs(p_grad - v_grad))
            divergence.append(local_div)
        
        return pd.Series(divergence, index=df.index)
    
    lyapunov_div = lyapunov_divergence(df)
    
    # Range Entropy Evolution
    def range_entropy(df, window=10):
        """Calculate entropy of price range distribution"""
        price_range = (df['high'] - df['low']) / df['close']
        
        entropy = price_range.rolling(window=window).apply(
            lambda x: -np.sum((np.histogram(x, bins=5)[0] / len(x)) * 
                            np.log(np.histogram(x, bins=5)[0] / len(x) + 1e-10))
        )
        return entropy
    
    range_entropy_signal = range_entropy(df)
    
    # Final Alpha Integration
    # Combine signals with multi-scale adaptive weighting
    signals = pd.DataFrame({
        'fractal_ratio': volume_fractal_ratio,
        'range_compression': range_fractal_compression,
        'fractal_accel': delta_fd_volume_3d,
        'fb_divergence': fb_divergence,
        'vol_concentration': vol_concentration,
        'order_persistence': order_persistence,
        'phase_coherence': phase_coherence_signal,
        'lead_lag': lead_lag,
        'lyapunov_div': lyapunov_div,
        'range_entropy': range_entropy_signal
    })
    
    # Multi-scale adaptive weighting
    short_term_weight = 0.4
    medium_term_weight = 0.35
    long_term_weight = 0.25
    
    # Classify signals by time horizon
    short_term_signals = ['fractal_accel', 'vol_concentration', 'lead_lag']
    medium_term_signals = ['fractal_ratio', 'range_compression', 'order_persistence', 'phase_coherence']
    long_term_signals = ['fb_divergence', 'lyapunov_div', 'range_entropy']
    
    # Normalize signals
    normalized_signals = signals.apply(lambda x: (x - x.rolling(window=50).mean()) / x.rolling(window=50).std())
    
    # Weighted combination
    short_component = normalized_signals[short_term_signals].mean(axis=1) * short_term_weight
    medium_component = normalized_signals[medium_term_signals].mean(axis=1) * medium_term_weight
    long_component = normalized_signals[long_term_signals].mean(axis=1) * long_term_weight
    
    # Final alpha factor
    fractal_alpha = short_component + medium_component + long_component
    
    return fractal_alpha
